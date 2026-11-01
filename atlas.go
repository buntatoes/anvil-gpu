// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato

package anvilgpu

// AtlasRegion is an integer texel rectangle inside a TextureAtlas.
type AtlasRegion struct {
	X, Y, Width, Height uint32
}

// TextureAtlas is a shelf-packed straight RGBA8 texture atlas.
type TextureAtlas struct {
	Width     uint32
	Height    uint32
	Pixels    []byte
	cursorX   uint32
	cursorY   uint32
	rowHeight uint32
}

// NewTextureAtlas allocates a zero-filled RGBA8 TextureAtlas.
func NewTextureAtlas(width, height uint32) *TextureAtlas {
	w := min(width, MaxSide)
	h := min(height, MaxSide)
	return &TextureAtlas{
		Width:  w,
		Height: h,
		Pixels: make([]byte, int(w)*int(h)*4),
	}
}

// Allocate shelf-packs rgba into the atlas and returns its AtlasRegion.
func (a *TextureAtlas) Allocate(width, height uint32, rgba []byte) (AtlasRegion, bool) {
	if a == nil || width == 0 || height == 0 || width > a.Width || height > a.Height || len(rgba) < int(width)*int(height)*4 {
		return AtlasRegion{}, false
	}
	if a.cursorX+width > a.Width {
		a.cursorX = 0
		a.cursorY += a.rowHeight + 1
		a.rowHeight = 0
	}
	if a.cursorY+height > a.Height {
		return AtlasRegion{}, false
	}
	region := AtlasRegion{
		X:      a.cursorX,
		Y:      a.cursorY,
		Width:  width,
		Height: height,
	}
	for row := uint32(0); row < height; row++ {
		srcStart := int(row*width) * 4
		dstStart := int((region.Y+row)*a.Width+region.X) * 4
		copy(a.Pixels[dstStart:dstStart+int(width)*4], rgba[srcStart:srcStart+int(width)*4])
	}
	a.cursorX += width + 1
	if height > a.rowHeight {
		a.rowHeight = height
	}
	return region, true
}

// SampleUV performs nearest-texel sampling of the atlas at normalized (u, v) in 0..1.
func (a *TextureAtlas) SampleUV(u, v float32) GpuColor {
	if a == nil || a.Width == 0 || a.Height == 0 || len(a.Pixels) < int(a.Width)*int(a.Height)*4 {
		return RGBA(255, 255, 255, 255)
	}
	px := min(uint32(roundHalfAwayFromZeroF32(clampF32(u, 0.0, 1.0)*float32(a.Width-1))), a.Width-1)
	py := min(uint32(roundHalfAwayFromZeroF32(clampF32(v, 0.0, 1.0)*float32(a.Height-1))), a.Height-1)
	idx := int(py*a.Width+px) * 4
	return RGBA(a.Pixels[idx], a.Pixels[idx+1], a.Pixels[idx+2], a.Pixels[idx+3])
}

// UVBounds returns the edge-normalized window [u0, v0, u1, v1] of region.
func (a *TextureAtlas) UVBounds(region AtlasRegion) [4]float32 {
	w := float32(max(a.Width, 1))
	h := float32(max(a.Height, 1))
	return [4]float32{
		float32(region.X) / w,
		float32(region.Y) / h,
		float32(region.X+region.Width) / w,
		float32(region.Y+region.Height) / h,
	}
}
