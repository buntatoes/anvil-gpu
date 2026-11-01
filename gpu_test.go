// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato

package anvilgpu

import (
	"bytes"
	"errors"
	"math"
	"strings"
	"testing"
)

func TestEmbeddedShadersPresent(t *testing.T) {
	if !strings.Contains(CompositorHLSL, "VSMain") || !strings.Contains(CompositorGLSL, "main") {
		t.Fatal("CompositorHLSL or CompositorGLSL missing expected entry points")
	}
	for _, ep := range PassEntryPoints {
		if !strings.Contains(PassesHLSL, ep.Entry) {
			t.Fatalf("PassesHLSL missing entry point %s", ep.Entry)
		}
	}
	for _, p := range GLSLPasses {
		if !strings.Contains(PassesGLSL, p) {
			t.Fatalf("PassesGLSL missing pass symbol %s", p)
		}
	}
}

func TestBT709AndNV12I420Surfaces(t *testing.T) {
	y, u, v := BT709RGBAToYUV(0, 0, 0)
	if y != 16 || u != 128 || v != 128 {
		t.Fatalf("black YUV = (%d,%d,%d), want (16,128,128)", y, u, v)
	}
	wy, wu, wv := BT709RGBAToYUV(255, 255, 255)
	if wy != 235 || wu != 128 || wv != 128 {
		t.Fatalf("white YUV = (%d,%d,%d), want (235,128,128)", wy, wu, wv)
	}
	c := BT709YUVToRGBA(235, 128, 128)
	if c != RGBA(255, 255, 255, 255) {
		t.Fatalf("white RGBA = %+v", c)
	}

	rgba := bytes.Repeat([]byte{200, 100, 50, 255}, 16)
	nv12, ok := Nv12FromRGBABT709(4, 4, rgba)
	if !ok {
		t.Fatal("Nv12FromRGBABT709 failed")
	}
	i420 := I420FromNv12(&nv12)
	if !bytes.Equal(nv12.ToRGBABT709(), i420.ToRGBABT709()) {
		t.Fatal("NV12 and I420 RGBA expansions differed")
	}
	backNv12 := i420.ToNv12()
	if !bytes.Equal(backNv12.UVPlane, nv12.UVPlane) {
		t.Fatal("I420.ToNv12 UVPlane mismatch")
	}
}

func TestTextureAtlasAndTiledGpuDevice(t *testing.T) {
	before := DeviceTilesShaded()
	quad := SolidQuad(NewGpuRect(2.0, 2.0, 20.0, 12.0), RGBA(20, 120, 220, 255)).
		WithRoundedBorder(4.0, 2.0, RGBA(255, 180, 40, 255))
	shadow := BoxShadowQuad(NewGpuRect(2.0, 2.0, 20.0, 12.0), 1.0, 2.0, 3.0, 0.0, 2.0, RGBA(0, 0, 0, 160))
	grad := LinearGradientQuad(NewGpuRect(0.0, 18.0, 30.0, 8.0), RGBA(255, 0, 0, 255), RGBA(0, 0, 255, 255), 90.0)
	quads := []GpuQuad{shadow, quad, grad}
	clearCol := RGBA(8, 8, 12, 255)
	dev := OpenGpuDevice()
	tiled := dev.RenderQuads(32, 28, clearCol, quads, nil)
	full := RenderFrameRGBA(32, 28, clearCol, quads, nil)
	if !bytes.Equal(tiled, full) {
		t.Fatal("GpuDevice.RenderQuads did not match RenderFrameRGBA")
	}
	if DeviceTilesShaded() <= before {
		t.Fatal("DeviceTilesShaded did not advance")
	}

	var src []byte
	for y := uint8(0); y < 4; y++ {
		for x := uint8(0); x < 6; x++ {
			src = append(src, x*40, y*60, 20, 255)
		}
	}
	drawn, ok := CompositeRGBAFrame(6, 4, src)
	if !ok || !bytes.Equal(drawn, src) {
		t.Fatal("CompositeRGBAFrame did not preserve source pixels")
	}
}

func TestCompositorPassesAndSceneBatch(t *testing.T) {
	pixels := make([]byte, 32*32*4)
	target, err := NewPackedRgbaTarget(pixels, 32, 32)
	if err != nil {
		t.Fatalf("NewPackedRgbaTarget failed: %v", err)
	}
	comp := OpenCompositor()

	// 1. Box shadow
	sh := NewShadowDesc(NewGpuRect(6, 6, 16, 12), RGBA(0, 0, 0, 200))
	sh.BlurRadius = 4.0
	sh.CornerRadius = 3.0
	if err := comp.BoxShadow(target, &sh); err != nil {
		t.Fatalf("BoxShadow failed: %v", err)
	}
	if px, _ := target.Pixel(14, 12); px[3] == 0 {
		t.Fatal("expected non-zero shadow alpha inside box")
	}

	// 2. Rounded gradient
	stops := []GpuGradientStop{
		NewGpuGradientStop(0.0, RGBA(255, 0, 0, 255)),
		NewGpuGradientStop(1.0, RGBA(0, 0, 255, 255)),
	}
	gDesc := GradientDesc{
		Rect:  NewGpuRect(4, 4, 24, 24),
		Shape: LinearGradientShape(4, 4, 28, 4),
		Stops: stops,
	}
	if err := comp.FillRoundedGradient(target, &gDesc, 4.0); err != nil {
		t.Fatalf("FillRoundedGradient failed: %v", err)
	}

	// 3. Blit NV12 (Bilinear)
	rgbaSrc := bytes.Repeat([]byte{40, 200, 80, 255}, 16)
	nv12, _ := Nv12FromRGBABT709(4, 4, rgbaSrc)
	vf := VideoFrameFromNv12(&nv12)
	bDesc := NewBlitDesc(8, 8, 8, 8)
	bDesc.Filter = FilterBilinear
	if err := comp.BlitFrame(target, &vf, &bDesc); err != nil {
		t.Fatalf("BlitFrame failed: %v", err)
	}
	if px, _ := target.Pixel(10, 10); px[1] < 150 {
		t.Fatalf("expected green-dominant blit pixel, got %v", px)
	}

	// 4. GpuSceneBatch execution
	batch := NewGpuSceneBatch()
	batch.PushClear(RGBA(10, 20, 30, 255))
	batch.PushClip(NewGpuRect(2, 2, 28, 28))
	batch.PushSolidRect(NewGpuRect(4, 4, 10, 10), RGBA(250, 120, 40, 255))
	batch.PushGlyphCoverage(6, 6, 2, 2, []byte{255, 128, 128, 255}, RGBA(255, 255, 255, 255))
	batch.PopClip()
	beforeBatches := SceneBatchesExecuted()
	if err := comp.ExecuteSceneBatch(target, batch); err != nil {
		t.Fatalf("ExecuteSceneBatch failed: %v", err)
	}
	if SceneBatchesExecuted() <= beforeBatches {
		t.Fatal("SceneBatchesExecuted did not increment")
	}
	if px, _ := target.Pixel(6, 6); px != [4]byte{255, 255, 255, 255} {
		t.Fatalf("expected glyph pixel (255,255,255,255), got %v", px)
	}
}

type failingBackend struct{}

func (failingBackend) Kind() BackendKind { return BackendHardware }
func (failingBackend) BoxShadow(target *RgbaTarget, _ *ShadowDesc) error {
	target.pxMut(0, 0)[0] = 99
	return ErrDevice
}
func (failingBackend) FillGradient(*RgbaTarget, *GradientDesc) error { return ErrDevice }
func (failingBackend) FillRoundedGradient(*RgbaTarget, *GradientDesc, float32) error {
	return ErrDevice
}
func (failingBackend) BlitFrame(*RgbaTarget, *VideoFrame, *BlitDesc) error { return ErrDevice }
func (failingBackend) ExecuteSceneBatch(*RgbaTarget, *GpuSceneBatch) error { return ErrDevice }

func TestCompositorTransactionalFallback(t *testing.T) {
	p1 := make([]byte, 16*16*4)
	p2 := make([]byte, 16*16*4)
	t1, _ := NewPackedRgbaTarget(p1, 16, 16)
	t2, _ := NewPackedRgbaTarget(p2, 16, 16)

	comp := CompositorWithDevice(failingBackend{})
	ref := SoftwareCompositor()
	sh := NewShadowDesc(NewGpuRect(0, 0, 8, 8), RGBA(200, 50, 50, 255))
	if err := comp.BoxShadow(t1, &sh); err != nil {
		t.Fatalf("fallback BoxShadow failed: %v", err)
	}
	_ = ref.BoxShadow(t2, &sh)
	if !bytes.Equal(p1, p2) {
		t.Fatal("transactional fallback did not restore and match software output")
	}
	if comp.Fallbacks() != 1 || comp.ActiveBackend() != BackendSoftware {
		t.Fatalf("unexpected fallback state: fallbacks=%d active=%v", comp.Fallbacks(), comp.ActiveBackend())
	}

	// Bad input returns ErrBadInput
	bad := NewShadowDesc(NewGpuRect(0, 0, -5, 8), RGBA(200, 50, 50, 255))
	if err := comp.BoxShadow(t1, &bad); !errors.Is(err, ErrBadInput) {
		t.Fatalf("expected ErrBadInput, got %v", err)
	}
}

func TestSecurityBoundaryAndAdversarialInputs(t *testing.T) {
	nan := float32(math.NaN())
	if got := clampF32(nan, 0.0, 1.0); got != 0.0 {
		t.Fatalf("clampF32(NaN) = %v, want 0.0", got)
	}

	// RenderFrameRGBA rejects oversized viewports and skips NaN/Inf quads without panicking.
	if fb := RenderFrameRGBA(MaxSide+1, 16, RGBA(0, 0, 0, 255), nil, nil); fb != nil {
		t.Fatal("expected nil framebuffer for width > MaxSide")
	}
	nanQuad := SolidQuad(NewGpuRect(nan, 0, 10, 10), RGBA(255, 0, 0, 255))
	fb := RenderFrameRGBA(8, 8, RGBA(10, 20, 30, 255), []GpuQuad{nanQuad}, nil)
	if len(fb) != 8*8*4 || fb[0] != 10 {
		t.Fatal("expected clean clear buffer when quad has NaN coordinates")
	}

	// Truncated or nil NV12 and I420 surfaces must not panic in ToRGBABT709 or ToNv12.
	var nilNV12 *Nv12Surface
	if nilNV12.ToRGBABT709() != nil {
		t.Fatal("expected nil from nil Nv12Surface.ToRGBABT709")
	}
	truncNV12 := Nv12Surface{Width: 16, Height: 16, YPlane: make([]byte, 4), UVPlane: make([]byte, 2)}
	if truncNV12.ToRGBABT709() != nil {
		t.Fatal("expected nil from truncated Nv12Surface.ToRGBABT709")
	}
	truncI420 := I420Surface{Width: 16, Height: 16, YPlane: make([]byte, 4), UPlane: make([]byte, 4), VPlane: make([]byte, 1)}
	if truncI420.ToRGBABT709() != nil {
		t.Fatal("expected nil from truncated I420Surface.ToRGBABT709")
	}
	_ = truncI420.ToNv12()

	// DrawGlyphCoverage with x = math.MaxInt32 must not overflow int32 into negative coordinates or panic.
	buf := make([]byte, 8*8*4)
	target, err := NewPackedRgbaTarget(buf, 8, 8)
	if err != nil {
		t.Fatalf("NewPackedRgbaTarget failed: %v", err)
	}
	cov := []byte{255, 255, 255, 255}
	if err := DrawGlyphCoverage(target, math.MaxInt32, math.MaxInt32, 2, 2, cov, RGBA(255, 255, 255, 255), nil); err != nil {
		t.Fatalf("DrawGlyphCoverage at MaxInt32 returned unexpected error: %v", err)
	}
	if err := DrawGlyphCoverage(target, 0, 0, MaxSide+1, 2, cov, RGBA(255, 255, 255, 255), nil); !errors.Is(err, ErrBadInput) {
		t.Fatalf("expected ErrBadInput on oversized glyph width, got %v", err)
	}
}
