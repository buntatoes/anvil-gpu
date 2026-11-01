// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato

package anvilgpu

// BT709RGBAToYUV converts 8-bit RGB to limited-range BT.709 YCbCr (Y in 16..235, Cb/Cr in 16..240).
func BT709RGBAToYUV(r, g, b uint8) (uint8, uint8, uint8) {
	rf := float32(r)
	gf := float32(g)
	bf := float32(b)
	y := uint8(clampF32(roundHalfAwayFromZeroF32(16.0+0.182586*rf+0.614231*gf+0.062007*bf), 16.0, 235.0))
	u := uint8(clampF32(roundHalfAwayFromZeroF32(128.0-0.100644*rf-0.338572*gf+0.439216*bf), 16.0, 240.0))
	v := uint8(clampF32(roundHalfAwayFromZeroF32(128.0+0.439216*rf-0.398942*gf-0.040274*bf), 16.0, 240.0))
	return y, u, v
}

// BT709YUVToRGBA converts limited-range BT.709 YCbCr to straight RGBA8 with alpha 255.
func BT709YUVToRGBA(y, u, v uint8) GpuColor {
	yf := maxF32(float32(y)-16.0, 0.0) * 1.164384
	uf := float32(u) - 128.0
	vf := float32(v) - 128.0
	r := uint8(clampF32(roundHalfAwayFromZeroF32(yf+1.792741*vf), 0.0, 255.0))
	g := uint8(clampF32(roundHalfAwayFromZeroF32(yf-0.213249*uf-0.532909*vf), 0.0, 255.0))
	b := uint8(clampF32(roundHalfAwayFromZeroF32(yf+2.112402*uf), 0.0, 255.0))
	return RGBA(r, g, b, 255)
}

// BT709YUVF32ToRGB converts limited-range BT.709 YCbCr code values in float32 to RGB8.
func BT709YUVF32ToRGB(y, u, v float32) [3]uint8 {
	yf := maxF32(y-16.0, 0.0) * 1.164384
	uf := u - 128.0
	vf := v - 128.0
	return [3]uint8{
		uint8(clampF32(roundHalfAwayFromZeroF32(yf+1.792741*vf), 0.0, 255.0)),
		uint8(clampF32(roundHalfAwayFromZeroF32(yf-0.213249*uf-0.532909*vf), 0.0, 255.0)),
		uint8(clampF32(roundHalfAwayFromZeroF32(yf+2.112402*uf), 0.0, 255.0)),
	}
}

// Nv12Surface is a semi-planar 4:2:0 NV12 buffer in limited-range BT.709.
type Nv12Surface struct {
	Width   uint32
	Height  uint32
	YPlane  []byte
	UVPlane []byte
}

// NewNv12Surface allocates an even-sized NV12 surface filled with limited-range black (Y=16, UV=128).
func NewNv12Surface(width, height uint32) Nv12Surface {
	w := min(max(width, 2), MaxSide) & ^uint32(1)
	h := min(max(height, 2), MaxSide) & ^uint32(1)
	yLen := int(w) * int(h)
	uvLen := int(w/2) * int(h/2) * 2
	yPlane := make([]byte, yLen)
	for i := range yPlane {
		yPlane[i] = 16
	}
	uvPlane := make([]byte, uvLen)
	for i := range uvPlane {
		uvPlane[i] = 128
	}
	return Nv12Surface{
		Width:   w,
		Height:  h,
		YPlane:  yPlane,
		UVPlane: uvPlane,
	}
}

// Nv12FromRGBABT709 packs tightly packed straight RGBA8 into limited-range BT.709 NV12.
func Nv12FromRGBABT709(width, height uint32, rgba []byte) (Nv12Surface, bool) {
	if width < 2 || height < 2 || width > MaxSide || height > MaxSide || width%2 != 0 || height%2 != 0 {
		return Nv12Surface{}, false
	}
	expected := int(width) * int(height) * 4
	if len(rgba) < expected {
		return Nv12Surface{}, false
	}
	surf := NewNv12Surface(width, height)
	w := int(width)
	h := int(height)
	uvW := w / 2
	for by := 0; by < h; by += 2 {
		for bx := 0; bx < w; bx += 2 {
			var uAcc, vAcc uint32
			for dy := 0; dy < 2; dy++ {
				for dx := 0; dx < 2; dx++ {
					px := bx + dx
					py := by + dy
					idx := (py*w + px) * 4
					y, u, v := BT709RGBAToYUV(rgba[idx], rgba[idx+1], rgba[idx+2])
					surf.YPlane[py*w+px] = y
					uAcc += uint32(u)
					vAcc += uint32(v)
				}
			}
			uvIdx := ((by/2)*uvW + (bx / 2)) * 2
			surf.UVPlane[uvIdx] = uint8((uAcc + 2) / 4)
			surf.UVPlane[uvIdx+1] = uint8((vAcc + 2) / 4)
		}
	}
	return surf, true
}

// ToRGBABT709 expands this NV12 surface to tightly packed straight RGBA8.
func (s *Nv12Surface) ToRGBABT709() []byte {
	if s == nil || s.Width < 2 || s.Height < 2 || s.Width > MaxSide || s.Height > MaxSide {
		return nil
	}
	w := int(s.Width)
	h := int(s.Height)
	uvW := (w + 1) / 2
	uvH := (h + 1) / 2
	if len(s.YPlane) < w*h || len(s.UVPlane) < uvW*uvH*2 {
		return nil
	}
	rgba := make([]byte, w*h*4)
	for py := 0; py < h; py++ {
		for px := 0; px < w; px++ {
			y := s.YPlane[py*w+px]
			uvIdx := ((py/2)*uvW + (px / 2)) * 2
			u := s.UVPlane[uvIdx]
			v := s.UVPlane[uvIdx+1]
			c := BT709YUVToRGBA(y, u, v)
			dst := (py*w + px) * 4
			rgba[dst] = c.R
			rgba[dst+1] = c.G
			rgba[dst+2] = c.B
			rgba[dst+3] = 255
		}
	}
	return rgba
}

// I420Surface is a planar 4:2:0 I420 (YUV420p) buffer in limited-range BT.709.
type I420Surface struct {
	Width  uint32
	Height uint32
	YPlane []byte
	UPlane []byte
	VPlane []byte
}

// NewI420Surface allocates an even-sized I420 surface filled with limited-range black (Y=16, U=128, V=128).
func NewI420Surface(width, height uint32) I420Surface {
	w := min(max(width, 2), MaxSide) & ^uint32(1)
	h := min(max(height, 2), MaxSide) & ^uint32(1)
	yLen := int(w) * int(h)
	cLen := int(w/2) * int(h/2)
	yPlane := make([]byte, yLen)
	for i := range yPlane {
		yPlane[i] = 16
	}
	uPlane := make([]byte, cLen)
	vPlane := make([]byte, cLen)
	for i := range uPlane {
		uPlane[i] = 128
		vPlane[i] = 128
	}
	return I420Surface{
		Width:  w,
		Height: h,
		YPlane: yPlane,
		UPlane: uPlane,
		VPlane: vPlane,
	}
}

// I420FromRGBABT709 converts tightly packed straight RGBA8 into limited-range BT.709 I420.
func I420FromRGBABT709(width, height uint32, rgba []byte) (I420Surface, bool) {
	nv12, ok := Nv12FromRGBABT709(width, height, rgba)
	if !ok {
		return I420Surface{}, false
	}
	return I420FromNv12(&nv12), true
}

// I420FromNv12 converts an Nv12Surface into planar I420Surface by de-interleaving UVPlane.
func I420FromNv12(nv12 *Nv12Surface) I420Surface {
	if nv12 == nil {
		return I420Surface{}
	}
	cLen := len(nv12.UVPlane) / 2
	uPlane := make([]byte, cLen)
	vPlane := make([]byte, cLen)
	for i := 0; i < cLen; i++ {
		uPlane[i] = nv12.UVPlane[2*i]
		vPlane[i] = nv12.UVPlane[2*i+1]
	}
	yCopy := make([]byte, len(nv12.YPlane))
	copy(yCopy, nv12.YPlane)
	return I420Surface{
		Width:  nv12.Width,
		Height: nv12.Height,
		YPlane: yCopy,
		UPlane: uPlane,
		VPlane: vPlane,
	}
}

// ToNv12 converts this planar I420Surface into a semi-planar Nv12Surface.
func (s *I420Surface) ToNv12() Nv12Surface {
	if s == nil {
		return Nv12Surface{}
	}
	cLen := min(len(s.UPlane), len(s.VPlane))
	uvPlane := make([]byte, cLen*2)
	for i := 0; i < cLen; i++ {
		uvPlane[2*i] = s.UPlane[i]
		uvPlane[2*i+1] = s.VPlane[i]
	}
	yCopy := make([]byte, len(s.YPlane))
	copy(yCopy, s.YPlane)
	return Nv12Surface{
		Width:   s.Width,
		Height:  s.Height,
		YPlane:  yCopy,
		UVPlane: uvPlane,
	}
}

// ToRGBABT709 expands this I420 surface into tightly packed straight RGBA8.
func (s *I420Surface) ToRGBABT709() []byte {
	if s == nil || s.Width < 2 || s.Height < 2 || s.Width > MaxSide || s.Height > MaxSide {
		return nil
	}
	w := int(s.Width)
	h := int(s.Height)
	cw := (w + 1) / 2
	ch := (h + 1) / 2
	if len(s.YPlane) < w*h || len(s.UPlane) < cw*ch || len(s.VPlane) < cw*ch {
		return nil
	}
	rgba := make([]byte, w*h*4)
	for py := 0; py < h; py++ {
		for px := 0; px < w; px++ {
			y := s.YPlane[py*w+px]
			cIdx := (py/2)*cw + (px / 2)
			u := s.UPlane[cIdx]
			v := s.VPlane[cIdx]
			c := BT709YUVToRGBA(y, u, v)
			dst := (py*w + px) * 4
			rgba[dst] = c.R
			rgba[dst+1] = c.G
			rgba[dst+2] = c.B
			rgba[dst+3] = 255
		}
	}
	return rgba
}
