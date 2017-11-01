package main

import (
	"github.com/xamber/Varis"
	"fmt"
)

func main() {
	n := varis.CreateNetwork(2, 3, 1)

	dataset := varis.Dataset{
		{[]float64{0.0, 0.0}, []float64{1.0}},
		{[]float64{1.0, 0.0}, []float64{0.0}},
		{[]float64{0.0, 1.0}, []float64{0.0}},
		{[]float64{1.0, 1.0}, []float64{1.0}},
	}

	varis.BackPropagation(&n, dataset, 10000)

	fmt.Println("After training")
	fmt.Println(0.0, 0.0, "-", n.Calculate(0.0, 0.0))
	fmt.Println(1.0, 0.0, "-", n.Calculate(1.0, 0.0))
	fmt.Println(0.0, 1.0, "-", n.Calculate(0.0, 1.0))
	fmt.Println(1.0, 1.0, "-", n.Calculate(1.0, 1.0))
}
