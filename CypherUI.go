package main

import (
	"encoding/json"
	"fmt"
	"image/color"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/theme"
	"fyne.io/fyne/v2/widget"

	"github.com/vova616/chipmunk"
	"github.com/vova616/chipmunk/vect"

	pb "Skynet/proto" // Your Protobuf ecosystem
)

// --- 1. STATE & TYPES ---

type ToxNetCore struct {
	App    fyne.App
	Window fyne.Window
	Vault  *pb.Vault
	Space  *chipmunk.Space

	Input   *widget.Entry
	ChatLog *widget.Entry // MultiLine for AI output
	Neurons []*Neuron
}

type Neuron struct {
	Shape *canvas.Circle
	Body  *chipmunk.Body
}

// --- 2. AI BRIDGE (The Brain Connector) ---

type AIResponse struct {
	Name    string   `json:"name"`
	Smiles  string   `json:"smiles"`
	Source  string   `json:"source"`
	Markers []string `json:"markers"`
	Error   string   `json:"error"`
}

func (t *ToxNetCore) callSkynetAI(query string) {
	t.appendChat("SYSTEM", "Analyzing molecular manifold for: "+query)

	go func() {
		// Calling the Python Skynet Bridge
		cmd := exec.Command("python3", "Skynet_Bridge.py", query)
		out, err := cmd.Output()

		fyne.CurrentApp().Driver().RunOnMain(func() {
			if err != nil {
				t.appendChat("SKYNET_ERROR", "Neural link failed. Ensure RDKit/PyTorch are active.")
				return
			}

			var resp AIResponse
			json.Unmarshal(out, &resp)

			if resp.Error != "" {
				t.appendChat("SKYNET", resp.Error)
				return
			}

			// Format the complex Biochemistry/Physics output
			header := fmt.Sprintf("[%s] Found via %s", resp.Name, resp.Source)
			t.appendChat("SKYNET", header)
			t.appendChat("STRUCTURE", "SMILES: "+resp.Smiles)
			
			if len(resp.Markers) > 0 {
				t.appendChat("RISK_ASSESSMENT", strings.Join(resp.Markers, " | "))
			} else {
				t.appendChat("RISK_ASSESSMENT", "No toxicological markers identified in latent space.")
			}
		})
	}()
}

func (t *ToxNetCore) appendChat(tag, msg string) {
	timestamp := time.Now().Format("15:04:05")
	current := t.ChatLog.Text
	newEntry := fmt.Sprintf("%s [%s] %s\n", timestamp, tag, msg)
	t.ChatLog.SetText(current + newEntry)
}

// --- 3. PHYSICS & DATA ---

func (t *ToxNetCore) LoadData() {
	path, _ := filepath.Abs("chemical_vault.bin")
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Println("Warning: Vault not found, AI will rely on live prediction.")
		return
	}
	t.Vault = &pb.Vault{}
	pb.Unmarshal(data, t.Vault)
}

func (t *ToxNetCore) InitPhysics() {
	t.Space = chipmunk.NewSpace()
	t.Space.Gravity = vect.Vect{X: 0, Y: 0}

	for i := 0; i < 60; i++ {
		radius := vect.Float(rand.Intn(4) + 2)
		shape := chipmunk.NewCircle(vect.Vect{0, 0}, radius)
		body := chipmunk.NewBody(1, shape.Moment(1))
		body.SetPosition(vect.Vect{vect.Float(rand.Intn(1600)), vect.Float(rand.Intn(1000))})
		body.SetVelocity(vect.Float(rand.Intn(60)-30), vect.Float(rand.Intn(60)-30))

		t.Space.AddBody(body)
		dot := canvas.NewCircle(color.NRGBA{0, 255, 150, 60})
		dot.Resize(fyne.NewSize(float32(radius*2), float32(radius*2)))
		t.Neurons = append(t.Neurons, &Neuron{Shape: dot, Body: body})
	}
}

// --- 4. THE UI ASSEMBLY ---

func (t *ToxNetCore) Assemble() fyne.CanvasObject {
	t.ChatLog = widget.NewMultiLineEntry()
	t.ChatLog.TextStyle = fyne.TextStyle{Monospace: true}
	t.ChatLog.Disable()

	t.Input = widget.NewEntry()
	t.Input.SetPlaceHolder("Enter Chemical Name or SMILES...")
	t.Input.OnSubmitted = func(s string) {
		if s == "" { return }
		t.appendChat("USER", s)
		t.callSkynetAI(s)
		t.Input.SetText("")
	}

	// Dynamic background
	bg := container.NewWithoutLayout()
	for _, n := range t.Neurons { bg.Add(n.Shape) }

	// Layout with a glass-morphism feel
	chatContainer := container.NewBorder(nil, t.Input, nil, nil, t.ChatLog)
	glassPanel := container.NewStack(
		canvas.NewRectangle(color.NRGBA{10, 10, 15, 200}),
		container.NewPadded(chatContainer),
	)

	return container.NewStack(bg, container.NewPadded(glassPanel))
}

func main() {
	core := &ToxNetCore{App: app.NewWithID("com.wasabi.toxnet")}
	core.LoadData()
	core.InitPhysics()

	core.Window = core.App.NewWindow("CYPHER_UI // SKYNET_v12")
	core.Window.SetContent(core.Assemble())
	core.Window.Resize(fyne.NewSize(1280, 720))

	// High-frequency physics loop
	go func() {
		ticker := time.NewTicker(time.Second / 60)
		for range ticker.C {
			core.Space.Step(1.0 / 60.0)
			for _, n := range core.Neurons {
				pos := n.Body.Position()
				n.Shape.Move(fyne.NewPos(float32(pos.X), float32(pos.Y)))
				// Boundary wrap-around
				if pos.X < 0 { n.Body.SetPosition(vect.Vect{1600, pos.Y}) }
				if pos.Y < 0 { n.Body.SetPosition(vect.Vect{pos.X, 1000}) }
				if pos.X > 1600 { n.Body.SetPosition(vect.Vect{0, pos.Y}) }
				if pos.Y > 1000 { n.Body.SetPosition(vect.Vect{pos.X, 0}) }
			}
			core.Window.Canvas().Refresh(core.Window.Content())
		}
	}()

	core.Window.ShowAndRun()
}