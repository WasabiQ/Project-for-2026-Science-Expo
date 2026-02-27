package main

import (
	"fmt"
	"image/color"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	// THE CORE ECOSYSTEM
	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/theme"
	"fyne.io/fyne/v2/widget"

	// PHYSICS: Chipmunk2D (The best 2D engine for Go)
	"github.com/vova616/chipmunk"
	"github.com/vova616/chipmunk/vect"

	// DATA: Google Protobuf
	pb "Skynet/proto"
)

// --- 1. THE ARCHITECTURAL STATE ---
type ToxNetCore struct {
	App    fyne.App
	Window fyne.Window
	Vault  *pb.Vault
	Space  *chipmunk.Space
	
	// UI Components (Pointer refs for reactive updates)
	Output  *widget.Entry
	Search  *widget.Entry
	Status  *canvas.Text
	Neurons []*Neuron
}

type Neuron struct {
	Shape *canvas.Circle
	Body  *chipmunk.Body
}

// --- 2. THE THEME ENGINE (Google Lab Aesthetic) ---
type labTheme struct{ font fyne.Resource }

func (t *labTheme) Color(n fyne.ThemeColorName, v fyne.ThemeVariant) color.Color {
	if n == theme.ColorNameBackground { return color.NRGBA{2, 4, 8, 255} }
	if n == theme.ColorNamePrimary { return color.NRGBA{0, 255, 150, 255} }
	return theme.DefaultTheme().Color(n, v)
}
func (t *labTheme) Font(s fyne.TextStyle) fyne.Resource { return t.font } // Hardcoded inject
func (t *labTheme) Icon(n fyne.ThemeIconName) fyne.Resource { return theme.DefaultTheme().Icon(n) }
func (t *labTheme) Size(n fyne.ThemeSizeName) float32      { return theme.DefaultTheme().Size(n) }

// --- 3. LOGIC MODULES (Minimalistic) ---

func (t *ToxNetCore) LoadData() {
	// Rely on OS library for relative path resolution
	path, _ := filepath.Abs("chemical_vault.bin")
	data, _ := os.ReadFile(path)
	t.Vault = &pb.Vault{}
	pb.Unmarshal(data, t.Vault)
}

func (t *ToxNetCore) InitPhysics() {
	t.Space = chipmunk.NewSpace()
	t.Space.Gravity = vect.Vect{X: 0, Y: 0} // Zero-G orbital movement

	// Create 100 library-managed bodies
	for i := 0; i < 100; i++ {
		radius := vect.Float(rand.Intn(3) + 2)
		shape := chipmunk.NewCircle(vect.Vect{0, 0}, radius)
		shape.SetElasticity(1)
		
		body := chipmunk.NewBody(1, shape.Moment(1))
		body.SetPosition(vect.Vect{vect.Float(rand.Intn(1200)), vect.Float(rand.Intn(800))})
		body.SetVelocity(vect.Float(rand.Intn(40)-20), vect.Float(rand.Intn(40)-20))
		
		t.Space.AddBody(body)
		t.Space.AddShape(shape)
		
		dot := canvas.NewCircle(color.NRGBA{0, 255, 150, 40})
		dot.Resize(fyne.NewSize(float32(radius*2), float32(radius*2)))
		t.Neurons = append(t.Neurons, &Neuron{Shape: dot, Body: body})
	}
}

// --- 4. THE UI COMPOSITION (The "Mac" Look) ---

func (t *ToxNetCore) Assemble() fyne.CanvasObject {
	t.Output = widget.NewMultiLineEntry()
	t.Output.Disable()
	
	t.Search = widget.NewEntry()
	t.Search.SetPlaceHolder("λ_Search_Molecular_ID...")
	t.Search.OnSubmitted = func(q string) {
		// Protobuf Map lookup is O(1) - Fast library logic
		if chem, ok := t.Vault.Entries[strings.ToLower(q)]; ok {
			t.Output.SetText(fmt.Sprintf("NAME: %s\nSMILES: %s\nMW: %.2f\nLOGP: %.2f", 
				chem.Name, chem.Smiles, chem.Descriptors.MolecularWeight, chem.Descriptors.Logp))
		}
	}

	// Tiling Layering
	bg := container.NewWithoutLayout()
	for _, n := range t.Neurons { bg.Add(n.Shape) }

	glassPanel := container.NewBorder(
		nil, container.NewPadded(t.Search), nil, nil, 
		container.NewStack(canvas.NewRectangle(color.NRGBA{255, 255, 255, 5}), t.Output),
	)

	return container.NewStack(bg, container.NewPadded(glassPanel))
}

// --- 5. EXECUTION ENGINE ---

func main() {
	core := &ToxNetCore{App: app.New()}
	core.LoadData()
	core.Window = core.App.NewWindow("TOXNET_TITAN")
	core.InitPhysics()

	core.Window.SetContent(core.Assemble())
	core.Window.SetFullScreen(true)

	// Library-based Ticker for 60FPS physics
	go func() {
		ticker := time.NewTicker(time.Second / 60)
		for range ticker.C {
			core.Space.Step(1.0 / 60.0)
			for _, n := range core.Neurons {
				pos := n.Body.Position()
				n.Shape.Move(fyne.NewPos(float32(pos.X), float32(pos.Y)))
			}
			core.Window.Canvas().Refresh(core.Window.Content())
		}
	}()

	core.Window.ShowAndRun()
}