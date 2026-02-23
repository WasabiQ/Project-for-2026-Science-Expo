package main

import (
	"bufio"
	"fmt"
	"image/color"
	"os/exec"
	"strings"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/widget"
)

func main() {
	myApp := app.New()
	window := myApp.NewWindow("ToxNet: Cypher Terminal")
	window.Resize(fyne.NewSize(900, 600))

	// --- 1. THE HEADER ---
	header := widget.NewLabel("TOXNET: CYPHER TERMINAL")
	header.Alignment = fyne.TextAlignCenter

	// --- 2. THE 150-NEURON VISUALISER (80-50-20) ---
	neuronGrid := container.NewWithoutLayout()
	drawNeurons(neuronGrid)
	neuronScroll := container.NewScroll(neuronGrid)
	neuronScroll.SetMinSize(fyne.NewSize(400, 300))
	neuronScroll.Hide() // Hidden by default

	// --- 3. OUTPUT AREA ---
	outputLabel := widget.NewLabel("Awaiting LHC Initialisation...")
	outputLabel.Wrapping = fyne.TextWrapWord

	// --- 4. THE LIVE EXECUTION ---
	inputField := widget.NewEntry()
	inputField.SetPlaceHolder("Enter 11 Markers (e.g., 0.1 0.5 ...)")

	scanBtn := widget.NewButton("INITIALISE SCAN", func() {
		outputLabel.SetText("BREACHING ATOMIC SHELL...")
		// Call the Python script
		res, score := runToxNetInference(inputField.Text)
		outputLabel.SetText(fmt.Sprintf("RESULT: %s\nIONISATION ENERGY: %s kcal/mol-1", res, score))
	})

	// --- 5. CONTROLS ---
	neuronBtn := widget.NewButton("Visible Neurons: OFF", func() {
		if neuronScroll.Visible() {
			neuronScroll.Hide()
			// neuronBtn.SetText("Visible Neurons: OFF")
		} else {
			neuronScroll.Show()
			// neuronBtn.SetText("Visible Neurons: ON")
		}
	})

	// LAYOUT ON THE "MOTHERBOARD"
	window.SetContent(container.NewVBox(
		header,
		inputField,
		container.NewHBox(scanBtn, neuronBtn),
		outputLabel,
		neuronScroll,
	))

	window.ShowAndRun()
}

// 150-Neuron Web Generation (80-50-20)
func drawNeurons(c *fyne.Container) {
	layers := []int{80, 50, 20}
	xOffset := float32(50)
	for _, count := range layers {
		for i := 0; i < count; i++ {
			dot := canvas.NewCircle(color.NRGBA{0, 255, 255, 150})
			dot.Resize(fyne.NewSize(4, 4))
			dot.Move(fyne.NewPos(xOffset, float32(i*8)))
			c.Add(dot)
		}
		xOffset += 100
	}
}

func runToxNetInference(data string) (string, string) {
	// Splitting data into the 11 markers
	markers := strings.Fields(data)
	// Args: ToxNet.py [Name] [SMILES] [Markers...]
	args := append([]string{"ToxNet.py", "SCAN_USER", "N/A"}, markers...)
	
	cmd := exec.Command("python", args...) // Use "python" or "python3" depending on your HP setup
	out, err := cmd.Output()
	if err != nil {
		return "ERROR", "0.000"
	}

	// Parsing: RESULT:TOXIC:0.99
	output := string(out)
	if strings.Contains(output, "RESULT") {
		parts := strings.Split(output, ":")
		return parts[1], parts[2]
	}
	return "UNKNOWN", "0.000"
}