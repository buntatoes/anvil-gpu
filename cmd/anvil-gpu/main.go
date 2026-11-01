// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato

package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"

	anvilgpu "github.com/buntatoes/anvil-gpu"
)

const (
	opRGBAToNV12    byte = 1
	opNV12ToRGBA    byte = 2
	opRGBAToI420    byte = 3
	opI420ToRGBA    byte = 4
	opRenderQuads   byte = 5
	opDrawBoxShadow byte = 6
	opDrawGradient  byte = 7
	opDrawBlitRGBA  byte = 8
	opDrawBlitNV12  byte = 9
	opDrawBlitI420  byte = 10
	opDrawSolidRect byte = 11
	opDrawGlyphCov  byte = 12

	maxWorkerBufferBytes = 64 * 1024 * 1024
)

func readF32(b []byte) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(b))
}

func readRect(b []byte) anvilgpu.GpuRect {
	return anvilgpu.NewGpuRect(readF32(b[0:4]), readF32(b[4:8]), readF32(b[8:12]), readF32(b[12:16]))
}

func readColor(b []byte) anvilgpu.GpuColor {
	return anvilgpu.RGBA(b[0], b[1], b[2], b[3])
}

func readOptClip(br *bufio.Reader) (*anvilgpu.GpuRect, error) {
	flag, err := br.ReadByte()
	if err != nil {
		return nil, err
	}
	if flag == 0 {
		return nil, nil
	}
	var buf [16]byte
	if _, err := io.ReadFull(br, buf[:]); err != nil {
		return nil, err
	}
	r := readRect(buf[:])
	return &r, nil
}

func runWorker(r io.Reader, w io.Writer) error {
	br := bufio.NewReaderSize(r, 256*1024)
	bw := bufio.NewWriterSize(w, 256*1024)
	var hdr [64]byte

	for {
		op, err := br.ReadByte()
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}

		switch op {
		case opRGBAToNV12:
			if _, err := io.ReadFull(br, hdr[:8]); err != nil {
				return err
			}
			width := binary.LittleEndian.Uint32(hdr[0:4])
			height := binary.LittleEndian.Uint32(hdr[4:8])
			if width > anvilgpu.MaxSide || height > anvilgpu.MaxSide {
				return anvilgpu.ErrBadTarget
			}
			rgba := make([]byte, int(width)*int(height)*4)
			if _, err := io.ReadFull(br, rgba); err != nil {
				return err
			}
			surf, ok := anvilgpu.ConvertRGBAOnDevice(width, height, rgba)
			if !ok {
				_ = bw.WriteByte(1)
			} else {
				_ = bw.WriteByte(0)
				binary.LittleEndian.PutUint32(hdr[0:4], surf.Width)
				binary.LittleEndian.PutUint32(hdr[4:8], surf.Height)
				binary.LittleEndian.PutUint32(hdr[8:12], uint32(len(surf.YPlane)))
				binary.LittleEndian.PutUint32(hdr[12:16], uint32(len(surf.UVPlane)))
				_, _ = bw.Write(hdr[:16])
				_, _ = bw.Write(surf.YPlane)
				_, _ = bw.Write(surf.UVPlane)
			}

		case opNV12ToRGBA:
			if _, err := io.ReadFull(br, hdr[:16]); err != nil {
				return err
			}
			width := binary.LittleEndian.Uint32(hdr[0:4])
			height := binary.LittleEndian.Uint32(hdr[4:8])
			yLen := binary.LittleEndian.Uint32(hdr[8:12])
			uvLen := binary.LittleEndian.Uint32(hdr[12:16])
			if width > anvilgpu.MaxSide || height > anvilgpu.MaxSide || yLen > maxWorkerBufferBytes || uvLen > maxWorkerBufferBytes {
				return anvilgpu.ErrBadTarget
			}
			yPlane := make([]byte, int(yLen))
			uvPlane := make([]byte, int(uvLen))
			if _, err := io.ReadFull(br, yPlane); err != nil {
				return err
			}
			if _, err := io.ReadFull(br, uvPlane); err != nil {
				return err
			}
			surf := anvilgpu.Nv12Surface{Width: width, Height: height, YPlane: yPlane, UVPlane: uvPlane}
			rgba := anvilgpu.PresentNV12OnDevice(&surf)
			if rgba == nil {
				_ = bw.WriteByte(byte(anvilgpu.ErrCodeBadInput))
			} else {
				_ = bw.WriteByte(0)
				binary.LittleEndian.PutUint32(hdr[0:4], uint32(len(rgba)))
				_, _ = bw.Write(hdr[:4])
				_, _ = bw.Write(rgba)
			}

		case opRGBAToI420:
			if _, err := io.ReadFull(br, hdr[:8]); err != nil {
				return err
			}
			width := binary.LittleEndian.Uint32(hdr[0:4])
			height := binary.LittleEndian.Uint32(hdr[4:8])
			if width > anvilgpu.MaxSide || height > anvilgpu.MaxSide {
				return anvilgpu.ErrBadTarget
			}
			rgba := make([]byte, int(width)*int(height)*4)
			if _, err := io.ReadFull(br, rgba); err != nil {
				return err
			}
			surf, ok := anvilgpu.I420FromRGBABT709(width, height, rgba)
			if !ok {
				_ = bw.WriteByte(1)
			} else {
				_ = bw.WriteByte(0)
				binary.LittleEndian.PutUint32(hdr[0:4], surf.Width)
				binary.LittleEndian.PutUint32(hdr[4:8], surf.Height)
				binary.LittleEndian.PutUint32(hdr[8:12], uint32(len(surf.YPlane)))
				binary.LittleEndian.PutUint32(hdr[12:16], uint32(len(surf.UPlane)))
				_, _ = bw.Write(hdr[:16])
				_, _ = bw.Write(surf.YPlane)
				_, _ = bw.Write(surf.UPlane)
				_, _ = bw.Write(surf.VPlane)
			}

		case opI420ToRGBA:
			if _, err := io.ReadFull(br, hdr[:16]); err != nil {
				return err
			}
			width := binary.LittleEndian.Uint32(hdr[0:4])
			height := binary.LittleEndian.Uint32(hdr[4:8])
			yLen := binary.LittleEndian.Uint32(hdr[8:12])
			cLen := binary.LittleEndian.Uint32(hdr[12:16])
			if width > anvilgpu.MaxSide || height > anvilgpu.MaxSide || yLen > maxWorkerBufferBytes || cLen > maxWorkerBufferBytes {
				return anvilgpu.ErrBadTarget
			}
			yPlane := make([]byte, int(yLen))
			uPlane := make([]byte, int(cLen))
			vPlane := make([]byte, int(cLen))
			if _, err := io.ReadFull(br, yPlane); err != nil {
				return err
			}
			if _, err := io.ReadFull(br, uPlane); err != nil {
				return err
			}
			if _, err := io.ReadFull(br, vPlane); err != nil {
				return err
			}
			surf := anvilgpu.I420Surface{Width: width, Height: height, YPlane: yPlane, UPlane: uPlane, VPlane: vPlane}
			rgba := surf.ToRGBABT709()
			if rgba == nil {
				_ = bw.WriteByte(byte(anvilgpu.ErrCodeBadInput))
			} else {
				_ = bw.WriteByte(0)
				binary.LittleEndian.PutUint32(hdr[0:4], uint32(len(rgba)))
				_, _ = bw.Write(hdr[:4])
				_, _ = bw.Write(rgba)
			}

		case opDrawBoxShadow:
			if _, err := io.ReadFull(br, hdr[:52]); err != nil {
				return err
			}
			width := binary.LittleEndian.Uint32(hdr[0:4])
			height := binary.LittleEndian.Uint32(hdr[4:8])
			stride := int(binary.LittleEndian.Uint32(hdr[8:12]))
			pLenU32 := binary.LittleEndian.Uint32(hdr[12:16])
			if pLenU32 > maxWorkerBufferBytes {
				return anvilgpu.ErrBadTarget
			}
			pLen := int(pLenU32)
			rect := readRect(hdr[16:32])
			corner := readF32(hdr[32:36])
			ox := readF32(hdr[36:40])
			oy := readF32(hdr[40:44])
			blur := readF32(hdr[44:48])
			spread := readF32(hdr[48:52])
			if _, err := io.ReadFull(br, hdr[:4]); err != nil {
				return err
			}
			color := readColor(hdr[:4])
			clip, err := readOptClip(br)
			if err != nil {
				return err
			}
			pixels := make([]byte, pLen)
			if _, err := io.ReadFull(br, pixels); err != nil {
				return err
			}
			target, err := anvilgpu.NewRgbaTarget(pixels, width, height, stride)
			if err != nil {
				_ = bw.WriteByte(byte(anvilgpu.ErrCodeBadTarget))
			} else {
				desc := anvilgpu.ShadowDesc{
					Rect:         rect,
					CornerRadius: corner,
					OffsetX:      ox,
					OffsetY:      oy,
					BlurRadius:   blur,
					Spread:       spread,
					Color:        color,
					Clip:         clip,
				}
				if err := anvilgpu.DrawBoxShadow(target, &desc); err != nil {
					_ = bw.WriteByte(byte(anvilgpu.ErrCodeBadInput))
				} else {
					_ = bw.WriteByte(0)
					_, _ = bw.Write(pixels)
				}
			}

		case opDrawSolidRect:
			if _, err := io.ReadFull(br, hdr[:48]); err != nil {
				return err
			}
			width := binary.LittleEndian.Uint32(hdr[0:4])
			height := binary.LittleEndian.Uint32(hdr[4:8])
			stride := int(binary.LittleEndian.Uint32(hdr[8:12]))
			pLenU32 := binary.LittleEndian.Uint32(hdr[12:16])
			if pLenU32 > maxWorkerBufferBytes {
				return anvilgpu.ErrBadTarget
			}
			pLen := int(pLenU32)
			rect := readRect(hdr[16:32])
			color := readColor(hdr[32:36])
			corner := readF32(hdr[36:40])
			bwVal := readF32(hdr[40:44])
			bColor := readColor(hdr[44:48])
			clip, err := readOptClip(br)
			if err != nil {
				return err
			}
			pixels := make([]byte, pLen)
			if _, err := io.ReadFull(br, pixels); err != nil {
				return err
			}
			target, err := anvilgpu.NewRgbaTarget(pixels, width, height, stride)
			if err != nil {
				_ = bw.WriteByte(byte(anvilgpu.ErrCodeBadTarget))
			} else if err := anvilgpu.DrawSolidRect(target, rect, color, corner, bwVal, bColor, clip); err != nil {
				_ = bw.WriteByte(byte(anvilgpu.ErrCodeBadInput))
			} else {
				_ = bw.WriteByte(0)
				_, _ = bw.Write(pixels)
			}

		default:
			return fmt.Errorf("unknown anvil-gpu worker opcode %d", op)
		}

		if err := bw.Flush(); err != nil {
			return err
		}
	}
}

func runSelfTest() error {
	comp := anvilgpu.OpenCompositor()
	buf := make([]byte, 16*16*4)
	target, err := anvilgpu.NewPackedRgbaTarget(buf, 16, 16)
	if err != nil {
		return err
	}
	batch := anvilgpu.NewGpuSceneBatch()
	batch.PushClear(anvilgpu.RGBA(12, 16, 22, 255))
	batch.PushRoundedRect(anvilgpu.NewGpuRect(2, 2, 12, 12), anvilgpu.RGBA(240, 100, 30, 255), 3.0, 1.0, anvilgpu.RGBA(255, 255, 255, 255))
	if err := comp.ExecuteSceneBatch(target, batch); err != nil {
		return err
	}
	px, ok := target.Pixel(8, 8)
	if !ok || px[0] != 240 {
		return errors.New("anvil-gpu self-test pixel mismatch")
	}
	return nil
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "--worker" {
		if err := runWorker(os.Stdin, os.Stdout); err != nil {
			os.Exit(1)
		}
		return
	}
	if err := runSelfTest(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("anvil-gpu %s ok\n", anvilgpu.Version)
}
