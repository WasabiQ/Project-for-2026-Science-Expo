package main

import (
	"fmt"
	"image/color"
	"os/exec"
	"runtime"
	"strings"
	"time"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/driver/desktop"
	"fyne.io/fyne/v2/layout"
	"fyne.io/fyne/v2/theme"
	"fyne.io/fyne/v2/widget"
)

// --- HOVER CONTROL ---
type hoverCtrl struct {
	widget.BaseWidget
	sidebar *fyne.Container
	blur    *canvas.Rectangle
}

func (h *hoverCtrl) MouseIn(*desktop.MouseEvent)  { h.sidebar.Show(); h.blur.Show(); h.Refresh() }
func (h *hoverCtrl) MouseOut(*desktop.MouseEvent) { h.sidebar.Hide(); h.blur.Hide(); h.Refresh() }

func main() {
	myApp := app.New()
	window := myApp.NewWindow("TOXNET: LHC REINFORCED")
	window.Resize(fyne.NewSize(1000, 750))

	// --- LHC BOOT SEQUENCE ---
	lhcLabel := widget.NewLabelWithStyle("STABILIZING HADRON COLLIDER...", fyne.TextAlignCenter, fyne.TextStyle{Monospace: true})
	progress := widget.NewProgressBar()
	bootScreen := container.NewVBox(layout.NewSpacer(), lhcLabel, progress, layout.NewSpacer())
	window.SetContent(bootScreen)

	// --- MAIN MENU COMPONENTS ---
	title := canvas.NewText("ToxNet", color.NRGBA{0, 255, 255, 255})
	title.TextSize = 85
	title.TextStyle = fyne.TextStyle{Bold: true, Italic: true}

	// NEURAL WEB (Neurons + Nerves)
	neuralWeb := container.NewWithoutLayout()
	drawNeuralWeb(neuralWeb)

	// --- SIDEBAR & SETTINGS ---
	blurLayer := canvas.NewRectangle(color.NRGBA{0, 0, 5, 220})
	blurLayer.Hide()

	vizCheck := widget.NewCheck("Visualize Neural Nerves", func(b bool) {
		if b { neuralWeb.Show() } else { neuralWeb.Hide() }
	})
	vizCheck.SetChecked(true)

	// Sidebar Content
	btnCSV := widget.NewButton("OPEN TOX21", func() { openData("Tox21.csv") })
	settings := container.NewVBox(
		widget.NewLabelWithStyle("CORE CONFIG", fyne.TextAlignCenter, fyne.TextStyle{Bold: true}),
		vizCheck,
		widget.NewSelect([]string{"System", "Light", "Dark"}, func(s string) {}),
		layout.NewSpacer(),
		btnCSV,
		widget.NewIcon(theme.SettingsIcon()),
	)
	sidebar := container.NewMax(canvas.NewRectangle(color.NRGBA{5, 5, 15, 255}), settings)
	sidebar.Hide()

	// --- EXECUTE BOOT ---
	go func() {
		for i := 0.0; i <= 1.0; i += 0.02 {
			progress.SetValue(i)
			time.Sleep(50 * time.Millisecond)
		}
		
		// Main Menu Layout
		mainUI := container.NewStack(
			canvas.NewRectangle(color.NRGBA{2, 2, 8, 255}),
			neuralWeb, // BACKGROUND WEB
			container.NewCenter(container.NewVBox(
				title,
				widget.NewButton("INITIALIZE SCAN", func() {}),
			)),
			blurLayer,
			container.NewHBox(sidebar, layout.NewSpacer()),
			&hoverCtrl{sidebar: sidebar, blur: blurLayer},
		)
		window.SetContent(mainUI)
	}()

	window.ShowAndRun()
}

// DRAWS NEURONS AND CONNECTING NERVES (LINES)
func drawNeuralWeb(c *fyne.Container) {
	layers := []int{80, 50, 20}
	var prevLayerPoints []fyne.Position

	xSpacing := float32(280)
	ySpacing := float32(8)

	for l, count := range layers {
		var currentLayerPoints []fyne.Position
		xPos := float32(l)*xSpacing + 100

		for i := 0; i < count; i++ {
			yPos := float32(i)*ySpacing + 40
			pos := fyne.NewPos(xPos, yPos)
			currentLayerPoints = append(currentLayerPoints, pos)

			// Draw Nerves (Connecting lines to previous layer)
			if l > 0 {
				for _, prevPos := range prevLayerPoints {
					// Only draw some nerves to avoid "Ruin" (clutter)
					if i%10 == 0 { 
						line := canvas.NewLine(color.NRGBA{0, 255, 255, 30}) // Very faint nerves
						line.Position1 = prevPos
						line.Position2 = pos
						line.StrokeWidth = 0.5
						c.Add(line)
					}
				}
			}

			// Draw Neuron (The Dot)
			dot := canvas.NewCircle(color.NRGBA{0, 255, 255, 180})
			dot.Resize(fyne.NewSize(3, 3))
			dot.Move(pos)
			c.Add(dot)
		}
		prevLayerPoints = currentLayerPoints
	}
}

func openData(f string) {
	switch runtime.GOOS {
	case "windows": exec.Command("cmd", "/c", "start", f).Start()
	case "darwin": exec.Command("open", f).Start()
	default: exec.Command("xdg-open", f).Start()
	}
}