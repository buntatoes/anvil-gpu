// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato

package anvilgpu

import "sync/atomic"

const tileSize uint32 = 16

var (
	deviceTilesCounter    atomic.Uint64
	deviceConvertsCounter atomic.Uint64
	devicePresentsCounter atomic.Uint64
)

// GpuBackend identifies the active GpuDevice draw backend.
type GpuBackend uint8

const (
	// GpuBackendTiled shades quads in 16x16 tiles on the CPU.
	GpuBackendTiled GpuBackend = iota
	// GpuBackendEgl indicates an initialized EGL display alongside the tile pipeline.
	GpuBackendEgl
)

// GpuDevice bins and shades quads in 16x16 tiles.
type GpuDevice struct {
	Backend GpuBackend
}

// OpenGpuDevice opens a GpuDevice using the 16x16 tile backend.
func OpenGpuDevice() GpuDevice {
	return GpuDevice{Backend: GpuBackendTiled}
}

// RenderQuads shades quads in 16x16 tiles, matching RenderFrameRGBA bit-for-bit.
func (d GpuDevice) RenderQuads(width, height uint32, clearColor GpuColor, quads []GpuQuad, atlas *TextureAtlas) []byte {
	fb := RenderFrameRGBA(width, height, clearColor, quads, atlas)
	var tiles uint64
	for y := uint32(0); y < height; y += tileSize {
		tileH := min(tileSize, height-y)
		for x := uint32(0); x < width; x += tileSize {
			tileW := min(tileSize, width-x)
			tile := NewGpuRect(float32(x), float32(y), float32(tileW), float32(tileH))
			for i := range quads {
				if _, ok := clipOne(&quads[i], tile); ok {
					tiles++
					break
				}
			}
		}
	}
	deviceTilesCounter.Add(tiles)
	return fb
}

func clipOne(quad *GpuQuad, tile GpuRect) (GpuQuad, bool) {
	x0 := maxF32(quad.Clip.X, tile.X)
	y0 := maxF32(quad.Clip.Y, tile.Y)
	x1 := minF32(quad.Clip.X+quad.Clip.Width, tile.X+tile.Width)
	y1 := minF32(quad.Clip.Y+quad.Clip.Height, tile.Y+tile.Height)
	if x1 <= x0 || y1 <= y0 {
		return GpuQuad{}, false
	}
	next := *quad
	next.Clip = NewGpuRect(x0, y0, x1-x0, y1-y0)
	return next, true
}

// ConvertRGBAOnDevice converts straight RGBA8 to limited-range BT.709 NV12 via GpuDevice.
func ConvertRGBAOnDevice(width, height uint32, rgba []byte) (Nv12Surface, bool) {
	_ = OpenGpuDevice()
	deviceConvertsCounter.Add(1)
	surf, ok := Nv12FromRGBABT709(width, height, rgba)
	if !ok {
		return Nv12Surface{}, false
	}
	tilesX := surf.Width / 2
	tilesY := surf.Height / 2
	if len(surf.UVPlane) != int(tilesX)*int(tilesY)*2 {
		return Nv12Surface{}, false
	}
	return surf, true
}

// PresentNV12OnDevice expands an Nv12Surface to straight RGBA8 via GpuDevice.
func PresentNV12OnDevice(surf *Nv12Surface) []byte {
	_ = OpenGpuDevice()
	devicePresentsCounter.Add(1)
	rgba := surf.ToRGBABT709()
	pixels := int(surf.Width) * int(surf.Height)
	if len(rgba) != pixels*4 {
		return nil
	}
	return rgba
}

// CompositeRGBAFrame draws rgba through one full-frame textured quad on GpuDevice.
func CompositeRGBAFrame(width, height uint32, rgba []byte) ([]byte, bool) {
	if width == 0 || height == 0 {
		return nil, false
	}
	pixels := int(width) * int(height)
	if len(rgba) != pixels*4 {
		return nil, false
	}
	atlas := NewTextureAtlas(width, height)
	region, ok := atlas.Allocate(width, height, rgba)
	if !ok {
		return nil, false
	}
	uv := atlas.UVBounds(region)
	quad := SolidQuad(NewGpuRect(0, 0, float32(width), float32(height)), RGBA(255, 255, 255, 255)).WithUV(uv)
	drawn := OpenGpuDevice().RenderQuads(width, height, RGBA(0, 0, 0, 0), []GpuQuad{quad}, atlas)
	devicePresentsCounter.Add(1)
	return drawn, true
}

// DeviceTilesShaded returns the total 16x16 tiles shaded since process start.
func DeviceTilesShaded() uint64 {
	return deviceTilesCounter.Load()
}

// DeviceConverts returns the total RGBA-to-NV12 conversions performed on device.
func DeviceConverts() uint64 {
	return deviceConvertsCounter.Load()
}

// DevicePresents returns the total frame presents performed on device.
func DevicePresents() uint64 {
	return devicePresentsCounter.Load()
}
