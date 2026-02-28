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
    "github.com/vova616/chipmunk"
    "github.com/vova616/chipmunk/vect"
    "fyne.io/fyne/v2/widget"
)

// --- 1. STATE & TYPES ---

type ToxNetCore struct {
    App    fyne.App
    Window fyne.Window
    // Vault  *pb.Vault  <-- Removed for the 8AM Emergency Fix
    Space  *chipmunk.Space

    Input   *widget.Entry
    ChatLog *widget.Entry 
    Neurons []*Neuron
}

type Neuron struct {
    Shape *canvas.Circle
    Body  *chipmunk.Body
}

// --- 2. AI BRIDGE ---

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

            t.appendChat("SKYNET", fmt.Sprintf("[%s] Found via %s", resp.Name, resp.Source))
            t.appendChat("STRUCTURE", "SMILES: "+resp.Smiles)
            
            if len(resp.Markers) > 0 {
                t.appendChat("RISK_ASSESSMENT", strings.Join(resp.Markers, " | "))
            } else {
                t.appendChat("RISK_ASSESSMENT", "No toxicological markers identified.")
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
    // Vault loading bypassed to remove Protobuf dependency for Expo launch
    fmt.Println("System: AI Bridge active. Local Vault bypassed for performance.")
}

func (t *ToxNetCore) InitPhysics() {
    t.Space = chipmunk.NewSpace()
    t.Space.Gravity = vect.Vect{X: 0, Y: 0}

    for i := 0; i < 60; i++ {
        radius := vect.Float(rand.Intn(4) + 2)
        shape := chipmunk.NewCircle(vect.Vect{0, 0}, radius)
        body := chipmunk.NewBody(1, shape.Moment(1))
        body.SetPosition(vect.Vect{vect.Float(rand.Intn(1200)), vect.Float(rand.Intn(700))})
        body.SetVelocity(vect.Float(rand.Intn(60)-30), vect.Float(rand.Intn(60)-30))

        t.Space.AddBody(body)
        dot := canvas.NewCircle(color.NRGBA{0, 255, 150, 100})
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

    bg := container.NewWithoutLayout()
    for _, n := range t.Neurons { bg.Add(n.Shape) }

    chatContainer := container.NewBorder(nil, t.Input, nil, nil, t.ChatLog)
    glassPanel := container.NewStack(
        canvas.NewRectangle(color.NRGBA{10, 10, 15, 220}),
        container.NewPadded(chatContainer),
    )

    return container.NewStack(bg, container.NewPadded(glassPanel))
}

func main() {
    core := &ToxNetCore{App: app.NewWithID("com.wasabi.toxnet")}
    core.LoadData()
    core.InitPhysics()

    core.Window = core.App.NewWindow("CYPHER_UI //