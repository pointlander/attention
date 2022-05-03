// Copyright 2022 The Attention Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/mjibson/go-dsp/fft"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/gradient/tf64"
)

func main() {
	Attention(1, false)
	Regular(1, false)
	Attention(1, true)
	Regular(1, true)

	attention := 0.0
	for i := 0; i < 128; i++ {
		attention += float64(Attention(int64(2+i), false))
	}
	attentionFFT := 0.0
	for i := 0; i < 128; i++ {
		attentionFFT += float64(Attention(int64(2+i), true))
	}
	regular := 0.0
	for i := 0; i < 128; i++ {
		regular += float64(Regular(int64(2+i), false))
	}
	regularFFT := 0.0
	for i := 0; i < 128; i++ {
		regularFFT += float64(Regular(int64(2+i), true))
	}
	fmt.Println("Attention:", attention/128)
	fmt.Println("Attention FFT:", attentionFFT/128)
	fmt.Println("Regular:", regular/128)
	fmt.Println("Regular FFT:", regularFFT/128)
}

// MakeLoops make worldline loops
func MakeWeightsFFT(rnd *rand.Rand, N int) []float64 {
	y := make([]complex128, 0, N)
	y = append(y, 0)
	for j := 1; j < N; j++ {
		y = append(y, complex(rnd.NormFloat64(), rnd.NormFloat64()))
	}
	yt := fft.FFT(y)

	/*min, max := math.MaxFloat64, -math.MaxFloat64
	for i := 0; i < N; i++ {
		r := real(yt[i])
		if r < min {
			min = r
		}
		if r > max {
			max = r
		}
	}

	norm := math.Abs(max - min)
	weights := make([]float64, 0, N)
	for i := 0; i < N; i++ {
		weights = append(weights, real(yt[i])/norm)
	}*/

	weights := make([]float64, 0, N)
	for i := 0; i < N; i++ {
		weights = append(weights, real(yt[i]))
	}

	return weights
}

// Attentino is an attention base neural network
func Attention(seed int64, fft bool) int {
	rnd := rand.New(rand.NewSource(seed))
	others := tf64.NewSet()
	others.Add("input", 2, 4)
	others.Add("output", 1, 4)

	w := others.Weights[0]
	w.X = append(w.X,
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	)
	w = others.Weights[1]
	w.X = append(w.X, 0, 1, 1, 0)

	set := tf64.NewSet()
	set.Add("aw", 2, 1)
	set.Add("bw", 2, 1)
	set.Add("ab", 1, 1)
	set.Add("bb", 1, 1)

	if fft {
		for _, w := range set.Weights[:2] {
			factor := math.Sqrt(2.0 / float64(w.S[0]))
			weights, weight := MakeWeightsFFT(rnd, cap(w.X)), 0
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, weights[weight]*factor)
				weight++
			}
		}
	} else {
		for _, w := range set.Weights[:2] {
			factor := math.Sqrt(2.0 / float64(w.S[0]))
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, rnd.NormFloat64()*factor)
			}
		}
	}

	for i := 2; i < len(set.Weights); i++ {
		set.Weights[i].X = set.Weights[i].X[:cap(set.Weights[i].X)]
	}

	deltas := make([][]float64, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float64, len(p.X)))
	}

	la := tf64.Add(tf64.Mul(set.Get("aw"), others.Get("input")), set.Get("ab"))
	lb := tf64.Add(tf64.Mul(set.Get("bw"), others.Get("input")), set.Get("bb"))
	f := tf64.Hadamard(la, lb)
	cost := tf64.Sum(tf64.Quadratic(f, others.Get("output")))

	alpha, eta, iterations := .3, .3, 1024
	points := make(plotter.XYs, 0, iterations)
	i := 0
	for i < iterations {
		total := 0.0
		set.Zero()

		total += tf64.Gradient(cost).X[0]
		sum := 0.0
		for _, p := range set.Weights[1:] {
			for _, d := range p.D {
				sum += d * d
			}
		}
		norm := math.Sqrt(sum)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}

		for j, w := range set.Weights {
			for k, d := range w.D {
				deltas[j][k] = alpha*deltas[j][k] - eta*d*scaling
				set.Weights[j].X[k] += deltas[j][k]
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: total})
		if seed == 1 {
			fmt.Println(i, total)
		}
		if total < 1e-6 {
			break
		}
		i++
	}

	if seed == 1 {
		f(func(a *tf64.V) bool {
			fmt.Println(i, a.X)
			return true
		})

		p := plot.New()

		p.Title.Text = "epochs vs cost"
		p.X.Label.Text = "epochs"
		p.Y.Label.Text = "cost"

		scatter, err := plotter.NewScatter(points)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		filename := "cost_attention.png"
		if fft {
			filename = "cost_attention_fft.png"
		}
		err = p.Save(8*vg.Inch, 8*vg.Inch, filename)
		if err != nil {
			panic(err)
		}
	}

	return i
}

// Regular is a regular neural network
func Regular(seed int64, fft bool) int {
	rnd := rand.New(rand.NewSource(seed))
	others := tf64.NewSet()
	others.Add("input", 2, 4)
	others.Add("output", 1, 4)

	w := others.Weights[0]
	w.X = append(w.X,
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	)
	w = others.Weights[1]
	w.X = append(w.X, 0, 1, 1, 0)

	set := tf64.NewSet()
	set.Add("aw", 2, 2)
	set.Add("bw", 2, 1)
	set.Add("ab", 2, 1)
	set.Add("bb", 1, 1)

	if fft {
		for _, w := range set.Weights[:2] {
			factor := math.Sqrt(2.0 / float64(w.S[0]))
			weights, weight := MakeWeightsFFT(rnd, cap(w.X)), 0
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, weights[weight]*factor)
				weight++
			}
		}
	} else {
		for _, w := range set.Weights[:2] {
			factor := math.Sqrt(2.0 / float64(w.S[0]))
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, rnd.NormFloat64()*factor)
			}
		}
	}

	for i := 2; i < len(set.Weights); i++ {
		set.Weights[i].X = set.Weights[i].X[:cap(set.Weights[i].X)]
	}

	deltas := make([][]float64, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float64, len(p.X)))
	}

	l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("aw"), others.Get("input")), set.Get("ab")))
	l2 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("bw"), l1), set.Get("bb")))
	cost := tf64.Sum(tf64.Quadratic(l2, others.Get("output")))

	alpha, eta, iterations := .3, .3, 1024
	points := make(plotter.XYs, 0, iterations)
	i := 0
	for i < iterations {
		total := 0.0
		set.Zero()

		total += tf64.Gradient(cost).X[0]
		sum := 0.0
		for _, p := range set.Weights[1:] {
			for _, d := range p.D {
				sum += d * d
			}
		}
		norm := math.Sqrt(sum)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}

		for j, w := range set.Weights {
			for k, d := range w.D {
				deltas[j][k] = alpha*deltas[j][k] - eta*d*scaling
				set.Weights[j].X[k] += deltas[j][k]
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: total})
		if seed == 1 {
			fmt.Println(i, total)
		}
		if total < 1e-6 {
			break
		}
		i++
	}

	if seed == 1 {
		l2(func(a *tf64.V) bool {
			fmt.Println(i, a.X)
			return true
		})

		p := plot.New()

		p.Title.Text = "epochs vs cost"
		p.X.Label.Text = "epochs"
		p.Y.Label.Text = "cost"

		scatter, err := plotter.NewScatter(points)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		filename := "cost_regular.png"
		if fft {
			filename = "cost_regular_fft.png"
		}
		err = p.Save(8*vg.Inch, 8*vg.Inch, filename)
		if err != nil {
			panic(err)
		}
	}

	return i
}
