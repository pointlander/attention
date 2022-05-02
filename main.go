// Copyright 2022 The Attention Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/gradient/tf64"
)

func main() {
	Attention(1)
	Regular(1)

	attention := 0.0
	for i := 0; i < 128; i++ {
		attention += float64(Attention(int64(2 + i)))
	}
	regular := 0.0
	for i := 0; i < 128; i++ {
		regular += float64(Regular(int64(2 + i)))
	}
	fmt.Println("Attention:", attention/128)
	fmt.Println("Regular:", regular/128)
}

// Attentino is an attention base neural network
func Attention(seed int64) int {
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

	for _, w := range set.Weights[:2] {
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, rnd.NormFloat64()*factor)
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

		err = p.Save(8*vg.Inch, 8*vg.Inch, "cost_attention.png")
		if err != nil {
			panic(err)
		}
	}

	return i
}

// Regular is a regular neural network
func Regular(seed int64) int {
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

	for _, w := range set.Weights[:2] {
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, rnd.NormFloat64()*factor)
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

		err = p.Save(8*vg.Inch, 8*vg.Inch, "cost_regular.png")
		if err != nil {
			panic(err)
		}
	}

	return i
}
