package varis

import (
	"fmt"
)

// Dataset - simple type for store input and expected Vectors.
type Dataset [][2]Vector

// PerceptronTrainer is a trainer for Perceptron networks
type PerceptronTrainer struct {
	Network *Perceptron
	Dataset Dataset
}

// BackPropagation train Network input Dataset for 'times' times.
func (t *PerceptronTrainer) BackPropagation(times int) error {
	var neuronDelta float64

	for iteration := 0; iteration < times; iteration++ {

		fmt.Println("Varis propogation iteraion ", iteration, " from ", times)

		for _, frame := range t.Dataset {
			expected := frame[1]
			results, err := t.Network.Calculate(frame[0])
			if err != nil {
				return err
			}

			layerDelta := 0.0
			for l := len(t.Network.layers) - 1; l > 0; l-- {
				nextLayerDelta := 0.00
				for i, n := range t.Network.layers[l] {
					if l == len(t.Network.layers)-1 {
						neuronDelta = (expected[i] - results[i]) * DEACTIVATION(n.getCore().cache)
					} else {
						neuronDelta = layerDelta * DEACTIVATION(n.getCore().cache)
					}
					neuronDelta *= float64(1)
					nextLayerDelta += neuronDelta
					n.changeWeight(neuronDelta)
				}
				layerDelta = nextLayerDelta
			}
		}
	}
	return nil
}
