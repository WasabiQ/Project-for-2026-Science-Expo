package main

import (
	"fmt"
	"image/color"
	"log"
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
	"google.golang.org/protobuf/proto"
	pb "Skynet/proto"
)

// --- 1. THE ARCHITECTURAL STATE ---
type ToxNetCore struct {
	App        fyne.App
	Window     fyne.Window
	Vault      *pb.Vault
	Space      *chipmunk.Space
	
	// UI Components (Pointer refs for reactive updates)
	Output     *widget.Entry
	Search     *widget.Entry
	Status     *canvas.Text
	Neurons    []*Neuron
	Background *container.Container
	stopTicker chan struct{}
}

type Neuron struct {
	Shape *canvas.Circle
	Body  *chipmunk.Body
}

// --- 2. THE THEME ENGINE (Google Lab Aesthetic) ---
type labTheme struct{ font fyne.Resource }

func (t *labTheme) Color(n fyne.ThemeColorName, v fyne.ThemeVariant) color.Color {
	if n == theme.ColorNameBackground { 
		return color.NRGBA{2, 4, 8, 255} 
	}
	if n == theme.ColorNamePrimary { 
		return color.NRGBA{0, 255, 150, 255} 
	}
	return theme.DefaultTheme().Color(n, v)
}

func (t *labTheme) Font(s fyne.TextStyle) fyne.Resource { 
	return t.font 
}

func (t *labTheme) Icon(n fyne.ThemeIconName) fyne.Resource { 
	return theme.DefaultTheme().Icon(n) 
}

func (t *labTheme) Size(n fyne.ThemeSizeName) float32 { 
	return theme.DefaultTheme().Size(n) 
}

// --- 3. LOGIC MODULES (Minimalistic) ---

func (t *ToxNetCore) LoadData() error {
	// Rely on OS library for relative path resolution
	path, err := filepath.Abs("chemical_vault.bin")
	if err != nil {
		log.Printf("Warning: Could not get absolute path: %v. Proceeding with empty vault.", err)
		t.Vault = &pb.Vault{Entries: make(map[string]*pb.ChemicalEntry)}
		return nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		log.Printf("Warning: Could not load chemical_vault.bin: %v. Proceeding with empty vault.", err)
		t.Vault = &pb.Vault{Entries: make(map[string]*pb.ChemicalEntry)}
		return nil
	}

	t.Vault = &pb.Vault{}
	
	// FIX: Use correct protobuf unmarshal syntax
	err = proto.Unmarshal(data, t.Vault)
	if err != nil {
		log.Printf("Warning: Could not unmarshal vault data: %v. Proceeding with empty vault.", err)
		t.Vault = &pb.Vault{Entries: make(map[string]*pb.ChemicalEntry)}
		return nil
	}

	log.Printf("Successfully loaded %d compounds from vault", len(t.Vault.Entries))
	return nil
}

func (t *ToxNetCore) InitPhysics() error {
	t.Space = chipmunk.NewSpace()
	t.Space.Gravity = vect.Vect{X: 0, Y: 0} // Zero-G orbital movement

	// Create 100 library-managed bodies
	rand.Seed(time.Now().UnixNano())
	
	for i := 0; i < 100; i++ {
		// FIX: Use float64 as intermediate type for correct conversion
		radius := float64(rand.Intn(3) + 2)
		shape := chipmunk.NewCircle(vect.Vect{0, 0}, vect.Float(radius))
		shape.SetElasticity(1)
		
		body := chipmunk.NewBody(1, shape.Moment(1))
		body.SetPosition(vect.Vect{
			X: vect.Float(rand.Intn(1200)),
			Y: vect.Float(rand.Intn(800)),
		})
		body.SetVelocity(
			vect.Float(rand.Intn(40)-20),
			vect.Float(rand.Intn(40)-20),
		)
		
		t.Space.AddBody(body)
		t.Space.AddShape(shape)
		
		dot := canvas.NewCircle(color.NRGBA{0, 255, 150, 40})
		// FIX: Proper type conversion for radius
		dot.Resize(fyne.NewSize(float32(radius*2), float32(radius*2)))
		t.Neurons = append(t.Neurons, &Neuron{Shape: dot, Body: body})
	}

	log.Printf("Physics engine initialized with %d neurons", len(t.Neurons))
	return nil
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
			output := fmt.Sprintf(
				"NAME: %s\nSMILES: %s\nMW: %.2f\nLOGP: %.2f",
				chem.Name,
				chem.Smiles,
				chem.Descriptors.MolecularWeight,
				chem.Descriptors.Logp,
			)
			t.Output.SetText(output)
		} else {
			t.Output.SetText(fmt.Sprintf("No entry found for: %s", q))
		}
	}

	// Tiling Layering - Background container for neurons
	t.Background = container.NewWithoutLayout()
	for _, n := range t.Neurons {
		t.Background.Add(n.Shape)
	}

	glassPanel := container.NewBorder(
		nil, 
		container.NewPadded(t.Search), 
		nil, 
		nil,
		container.NewStack(
			canvas.NewRectangle(color.NRGBA{255, 255, 255, 5}),
			t.Output,
		),
	)

	return container.NewStack(t.Background, container.NewPadded(glassPanel))
}

// --- 5. EXECUTION ENGINE ---

func main() {
	core := &ToxNetCore{
		App:        app.New(),
		stopTicker: make(chan struct{}),
	}

	// Load data with proper error handling
	if err := core.LoadData(); err != nil {
		log.Fatalf("Failed to initialize: %v", err)
	}

	core.Window = core.App.NewWindow("TOXNET_TITAN")
	
	// Initialize physics with error handling
	if err := core.InitPhysics(); err != nil {
		log.Fatalf("Failed to initialize physics: %v", err)
	}

	core.Window.SetContent(core.Assemble())
	core.Window.SetFullScreen(true)

	// FIX: Efficient physics loop with proper resource cleanup
	go func() {
		ticker := time.NewTicker(time.Second / 60)
		defer ticker.Stop()
		
		for {
			select {
			case <-ticker.C:
				core.Space.Step(1.0 / 60.0)
				
				// Update neuron positions only (more efficient than full refresh)
				for _, n := range core.Neurons {
					pos := n.Body.Position()
					n.Shape.Move(fyne.NewPos(float32(pos.X), float32(pos.Y)))
				}
				
				// FIX: Refresh only the background canvas, not entire window
				core.Background.Refresh()
				
			case <-core.stopTicker:
				return
			}
		}
	}()

	core.Window.ShowAndRun()
	close(core.stopTicker)
}
