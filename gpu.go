// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Buntato

// Package anvilgpu implements 2D compositor passes (blurred box shadows, multi-stop
// linear/radial/rounded gradients, solid/bordered SDF rounded boxes, glyph coverage,
// and scaled RGBA/NV12/I420 video frame blits), batched scene execution (GpuSceneBatch),
// 16x16 tiled quad rasterization (GpuDevice), shelf-packed RGBA8 texture atlases, and
// limited-range BT.709 YCbCr (Nv12Surface / I420Surface) conversions.
package anvilgpu

import (
	"math"
	"sync/atomic"
)

// Version is the semantic version of anvil-gpu.
const Version = "0.12.0"

var (
	devicePassDrawsCounter      atomic.Uint64
	sceneBatchesExecutedCounter atomic.Uint64
	sceneOpsExecutedCounter     atomic.Uint64
)

// DevicePassDraws returns the total number of compositor pass draws recorded.
func DevicePassDraws() uint64 {
	return devicePassDrawsCounter.Load()
}

// SceneBatchesExecuted returns the total number of GpuSceneBatch executions recorded.
func SceneBatchesExecuted() uint64 {
	return sceneBatchesExecutedCounter.Load()
}

// SceneOpsExecuted returns the total number of individual GpuSceneOp items executed across batches.
func SceneOpsExecuted() uint64 {
	return sceneOpsExecutedCounter.Load()
}

func recordSceneBatchTelemetry(opCount int) {
	sceneBatchesExecutedCounter.Add(1)
	sceneOpsExecutedCounter.Add(uint64(opCount))
}

// GpuColor represents an 8-bit straight (unpremultiplied) RGBA color.
type GpuColor struct {
	R, G, B, A uint8
}

// RGBA constructs a straight GpuColor.
func RGBA(r, g, b, a uint8) GpuColor {
	return GpuColor{R: r, G: g, B: b, A: a}
}

// GpuRect is an axis-aligned rectangle where (X, Y) is the minimum corner.
type GpuRect struct {
	X, Y, Width, Height float32
}

// NewGpuRect constructs a GpuRect from minimum corner and size.
func NewGpuRect(x, y, width, height float32) GpuRect {
	return GpuRect{X: x, Y: y, Width: width, Height: height}
}

// Contains reports whether (px, py) lies inside the rectangle (exclusive of far edges).
func (r GpuRect) Contains(px, py float32) bool {
	return px >= r.X && py >= r.Y && px < r.X+r.Width && py < r.Y+r.Height
}

// GpuGradientStop is one color stop on a multi-stop linear or radial gradient ramp.
type GpuGradientStop struct {
	Offset float32
	Color  GpuColor
}

// NewGpuGradientStop constructs a gradient stop at offset (0..1).
func NewGpuGradientStop(offset float32, color GpuColor) GpuGradientStop {
	return GpuGradientStop{Offset: offset, Color: color}
}

// QuadKind describes how a GpuQuad fills its rectangle.
type QuadKind uint8

const (
	// QuadKindSolid fills with flat FillColor.
	QuadKindSolid QuadKind = iota
	// QuadKindLinearGradient fills with a two-stop linear gradient from FillColor to EndColor.
	QuadKindLinearGradient
	// QuadKindBoxShadow draws a soft SDF box shadow with BlurRadius.
	QuadKindBoxShadow
)

// GpuQuad is one quad rasterized by RenderFrameRGBA or GpuDevice.RenderQuads.
type GpuQuad struct {
	Rect         GpuRect
	Clip         GpuRect
	FillColor    GpuColor
	BorderColor  GpuColor
	CornerRadius float32
	BorderWidth  float32
	HasUV        bool
	UVRect       [4]float32
	Kind         QuadKind
	EndColor     GpuColor
	AngleDeg     float32
	BlurRadius   float32
}

// SolidQuad constructs a solid-filled GpuQuad whose clip matches rect.
func SolidQuad(rect GpuRect, fill GpuColor) GpuQuad {
	return GpuQuad{
		Rect:      rect,
		Clip:      rect,
		FillColor: fill,
		Kind:      QuadKindSolid,
	}
}

// LinearGradientQuad constructs a two-stop linear gradient GpuQuad across rect.
func LinearGradientQuad(rect GpuRect, start, end GpuColor, angleDeg float32) GpuQuad {
	return GpuQuad{
		Rect:      rect,
		Clip:      rect,
		FillColor: start,
		Kind:      QuadKindLinearGradient,
		EndColor:  end,
		AngleDeg:  angleDeg,
	}
}

// BoxShadowQuad constructs a shadow GpuQuad expanded by spread and offset.
func BoxShadowQuad(boxRect GpuRect, offsetX, offsetY, blurRadius, spreadRadius, cornerRadius float32, color GpuColor) GpuQuad {
	blur := maxF32(blurRadius, 0.0)
	spread := spreadRadius
	rect := NewGpuRect(
		boxRect.X+offsetX-spread,
		boxRect.Y+offsetY-spread,
		maxF32(boxRect.Width+spread*2.0, 1.0),
		maxF32(boxRect.Height+spread*2.0, 1.0),
	)
	clip := NewGpuRect(
		rect.X-blur,
		rect.Y-blur,
		rect.Width+blur*2.0,
		rect.Height+blur*2.0,
	)
	return GpuQuad{
		Rect:         rect,
		Clip:         clip,
		FillColor:    color,
		CornerRadius: maxF32(cornerRadius+spread, 0.0),
		Kind:         QuadKindBoxShadow,
		BlurRadius:   blur,
	}
}

// WithRoundedBorder returns a copy of q with rounded border radius, stroke width, and color.
func (q GpuQuad) WithRoundedBorder(cornerRadius, borderWidth float32, borderColor GpuColor) GpuQuad {
	q.CornerRadius = maxF32(cornerRadius, 0.0)
	q.BorderWidth = maxF32(borderWidth, 0.0)
	q.BorderColor = borderColor
	return q
}

// WithClip returns a copy of q with its clip rectangle replaced.
func (q GpuQuad) WithClip(clip GpuRect) GpuQuad {
	q.Clip = clip
	return q
}

// WithUV returns a copy of q with an attached texture atlas UV window.
func (q GpuQuad) WithUV(uv [4]float32) GpuQuad {
	q.HasUV = true
	q.UVRect = uv
	return q
}

func clampF32(v, lo, hi float32) float32 {
	if math.IsNaN(float64(v)) {
		return lo
	}
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func minF32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func maxF32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func roundHalfAwayFromZeroF32(v float32) float32 {
	return float32(math.Round(float64(v)))
}

func floorF32(v float32) float32 {
	return float32(math.Floor(float64(v)))
}

func ceilF32(v float32) float32 {
	return float32(math.Ceil(float64(v)))
}

// SDFRoundedBox computes the signed distance from (relX, relY) (relative to box center)
// to a rounded rectangle with half-extents (halfW, halfH) and corner radius.
func SDFRoundedBox(relX, relY, halfW, halfH, radius float32) float32 {
	r := maxF32(minF32(minF32(radius, halfW), halfH), 0.0)
	qx := float32(math.Abs(float64(relX))) - (halfW - r)
	qy := float32(math.Abs(float64(relY))) - (halfH - r)
	ox := maxF32(qx, 0.0)
	oy := maxF32(qy, 0.0)
	outside := float32(math.Sqrt(float64(ox*ox + oy*oy)))
	inside := minF32(maxF32(qx, qy), 0.0)
	return outside + inside - r
}

// SDFBoxShadowAlpha computes the Hermite smoothstep shadow coverage at signed distance dist.
func SDFBoxShadowAlpha(dist, blurRadius float32) float32 {
	sigma := maxF32(blurRadius, 0.5)
	t := clampF32(0.5-dist/sigma, 0.0, 1.0)
	return t * t * (3.0 - 2.0*t)
}

// SampleLinearGradient samples a two-stop linear gradient in box-center space.
func SampleLinearGradient(relX, relY, halfW, halfH, angleDeg float32, start, end GpuColor) GpuColor {
	rad := float64(angleDeg) * math.Pi / 180.0
	dx := float32(math.Sin(rad))
	dy := -float32(math.Cos(rad))
	maxExtent := maxF32(float32(math.Abs(float64(dx)))*halfW+float32(math.Abs(float64(dy)))*halfH, 0.5)
	proj := relX*dx + relY*dy
	t := clampF32((proj/maxExtent)*0.5+0.5, 0.0, 1.0)
	return LerpColor(start, end, t)
}

// SampleMultiStopGradient samples a multi-stop color ramp at normalized parameter t (0..1).
func SampleMultiStopGradient(stops []GpuGradientStop, t float32) GpuColor {
	if len(stops) == 0 {
		return RGBA(0, 0, 0, 0)
	}
	if len(stops) == 1 {
		return stops[0].Color
	}
	clamped := clampF32(t, 0.0, 1.0)
	if clamped <= stops[0].Offset {
		return stops[0].Color
	}
	for i := 0; i+1 < len(stops); i++ {
		a := stops[i]
		b := stops[i+1]
		if clamped <= b.Offset {
			span := maxF32(b.Offset-a.Offset, 1e-5)
			localT := clampF32((clamped-a.Offset)/span, 0.0, 1.0)
			return LerpColor(a.Color, b.Color, localT)
		}
	}
	return stops[len(stops)-1].Color
}

// SampleRadialGradient samples an elliptical radial gradient centered at (cx, cy) with radii (rx, ry).
func SampleRadialGradient(cx, cy, rx, ry, px, py float32, stops []GpuGradientStop) GpuColor {
	nx := (px - cx) / maxF32(rx, 0.5)
	ny := (py - cy) / maxF32(ry, 0.5)
	dist := clampF32(float32(math.Sqrt(float64(nx*nx+ny*ny))), 0.0, 1.0)
	return SampleMultiStopGradient(stops, dist)
}

// ModulateColor multiplies two straight colors channel-wise with rounding (a*b + 127) / 255.
func ModulateColor(a, b GpuColor) GpuColor {
	return GpuColor{
		R: uint8((uint16(a.R)*uint16(b.R) + 127) / 255),
		G: uint8((uint16(a.G)*uint16(b.G) + 127) / 255),
		B: uint8((uint16(a.B)*uint16(b.B) + 127) / 255),
		A: uint8((uint16(a.A)*uint16(b.A) + 127) / 255),
	}
}

// LerpColor linearly interpolates two straight colors with floor(x + 0.5) rounding.
func LerpColor(a, b GpuColor, t float32) GpuColor {
	inv := 1.0 - t
	return GpuColor{
		R: uint8(floorF32(float32(a.R)*inv + float32(b.R)*t + 0.5)),
		G: uint8(floorF32(float32(a.G)*inv + float32(b.G)*t + 0.5)),
		B: uint8(floorF32(float32(a.B)*inv + float32(b.B)*t + 0.5)),
		A: uint8(floorF32(float32(a.A)*inv + float32(b.A)*t + 0.5)),
	}
}

// BlendPixel composites src over a 4-byte destination pixel using straight alpha and forces dst[3]=255.
func BlendPixel(dst []byte, src GpuColor) {
	if src.A == 255 {
		dst[0] = src.R
		dst[1] = src.G
		dst[2] = src.B
		dst[3] = 255
		return
	}
	if src.A == 0 {
		return
	}
	alpha := uint16(src.A)
	inv := 255 - alpha
	dst[0] = uint8((uint16(src.R)*alpha + uint16(dst[0])*inv + 127) / 255)
	dst[1] = uint8((uint16(src.G)*alpha + uint16(dst[1])*inv + 127) / 255)
	dst[2] = uint8((uint16(src.B)*alpha + uint16(dst[2])*inv + 127) / 255)
	dst[3] = 255
}

// RenderFrameRGBA rasterizes quads in painter's order into a straight RGBA8 buffer.
func RenderFrameRGBA(width, height uint32, clearColor GpuColor, quads []GpuQuad, atlas *TextureAtlas) []byte {
	if width == 0 || height == 0 || width > MaxSide || height > MaxSide {
		return nil
	}
	fb := make([]byte, int(width)*int(height)*4)
	for i := 0; i < len(fb); i += 4 {
		fb[i] = clearColor.R
		fb[i+1] = clearColor.G
		fb[i+2] = clearColor.B
		fb[i+3] = clearColor.A
	}

	for _, quad := range quads {
		if !rectIsValid(quad.Rect) || !rectIsValid(quad.Clip) {
			continue
		}
		var pad float32
		if quad.Kind == QuadKindBoxShadow {
			pad = maxF32(quad.BlurRadius, 0.0)
		}
		minX := uint32(maxF32(floorF32(maxF32(quad.Rect.X-pad, quad.Clip.X)), 0.0))
		minY := uint32(maxF32(floorF32(maxF32(quad.Rect.Y-pad, quad.Clip.Y)), 0.0))
		maxX := uint32(minF32(ceilF32(minF32(quad.Rect.X+quad.Rect.Width+pad, quad.Clip.X+quad.Clip.Width)), float32(width)))
		maxY := uint32(minF32(ceilF32(minF32(quad.Rect.Y+quad.Rect.Height+pad, quad.Clip.Y+quad.Clip.Height)), float32(height)))

		halfW := quad.Rect.Width * 0.5
		halfH := quad.Rect.Height * 0.5
		cx := quad.Rect.X + halfW
		cy := quad.Rect.Y + halfH

		for py := minY; py < maxY; py++ {
			for px := minX; px < maxX; px++ {
				sampleX := float32(px) + 0.5
				sampleY := float32(py) + 0.5
				relX := sampleX - cx
				relY := sampleY - cy
				dist := SDFRoundedBox(relX, relY, halfW, halfH, quad.CornerRadius)

				if quad.Kind == QuadKindBoxShadow {
					shadowAlpha := SDFBoxShadowAlpha(dist, quad.BlurRadius)
					if shadowAlpha <= 0.0 {
						continue
					}
					color := quad.FillColor
					color.A = uint8(roundHalfAwayFromZeroF32(float32(color.A) * shadowAlpha))
					dstIdx := int(py*width+px) * 4
					BlendPixel(fb[dstIdx:dstIdx+4], color)
					continue
				}

				outerAlpha := clampF32(0.5-dist, 0.0, 1.0)
				if outerAlpha <= 0.0 {
					continue
				}

				color := quad.FillColor
				if quad.Kind == QuadKindLinearGradient {
					color = SampleLinearGradient(relX, relY, halfW, halfH, quad.AngleDeg, quad.FillColor, quad.EndColor)
				}
				if quad.HasUV && atlas != nil {
					tx := clampF32((sampleX-quad.Rect.X)/maxF32(quad.Rect.Width, 1.0), 0.0, 1.0)
					ty := clampF32((sampleY-quad.Rect.Y)/maxF32(quad.Rect.Height, 1.0), 0.0, 1.0)
					u := quad.UVRect[0] + (quad.UVRect[2]-quad.UVRect[0])*tx
					v := quad.UVRect[1] + (quad.UVRect[3]-quad.UVRect[1])*ty
					sampled := atlas.SampleUV(u, v)
					color = ModulateColor(color, sampled)
				}
				if quad.BorderWidth > 0.0 {
					innerDist := dist + quad.BorderWidth
					borderMix := clampF32(0.5+innerDist, 0.0, 1.0)
					color = LerpColor(color, quad.BorderColor, borderMix)
				}
				color.A = uint8(roundHalfAwayFromZeroF32(float32(color.A) * outerAlpha))
				dstIdx := int(py*width+px) * 4
				BlendPixel(fb[dstIdx:dstIdx+4], color)
			}
		}
	}
	return fb
}

// D3D11HardwareStatus describes the Direct3D 11 / CPU tile compositor status.
type D3D11HardwareStatus struct {
	DeviceCreated bool
	FeatureLevel  uint32
	VSBytecodeLen int
	PSBytecodeLen int
	BackendLabel  string
}

// ProbeAndCompileD3D11Compositor reports the active compositor status.
func ProbeAndCompileD3D11Compositor() D3D11HardwareStatus {
	return D3D11HardwareStatus{
		DeviceCreated: false,
		FeatureLevel:  0,
		VSBytecodeLen: len(CompositorHLSL),
		PSBytecodeLen: len(PassesHLSL),
		BackendLabel:  "CPU Tile Compositor Fallback",
	}
}
