import trainer
import torch
import tkinter as tk

model = trainer.myLSTM(trainer.vocab_size, trainer.hidden_size, trainer.num_layers).to(trainer.device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

storedOut = ""
def generate_text():
    str = input_var.get()
    len = int(length_var.get())
    t = float(temperature_var.get())

    global storedOut

    # Reuse the generate_text function from trainer.py
    text = trainer.generate_text(model, start_string=str, length=len, temp=t) 
        # temp = 0.15: lowest temp without something breaking
        # temp = 0.4-0.6: a safe range though it gets repetitive at lower ends.
        # temp = 0.6-0.8:  basically ideal. Still bad at paragraphs, marginally better for sentences, but good at words.
        # temp = 1: breakdown kinda begins.
        # temp = 1.2 - words sometimes make sense. Sentence structure and names maintained. line breaks becoming more uneven in spacing
        # temp = 2 -  some hints of coherence are still present but it is for the large part useless
    
    storedOut = text 
    print(storedOut)

 
root = tk.Tk()
root.title("tolst-AI")
root.geometry("400x300")
root.resizable(False, False)
#a = tk.Label(root, text="tolst-AI v0.1", wraplength=350)
#a.pack()


input_var = tk.StringVar(value="the")
length_var = tk.StringVar(value="250")
temperature_var = tk.StringVar(value="0.65")

tk.Label(root, text = "Input Str:").grid(row=0, sticky="e", padx=5, pady=5)
b = tk.Entry(root, text="input_string", textvariable=input_var)
tk.Label(root, text = "Temp:").grid(row=1, sticky="e", padx=5, pady=5)
t = tk.Entry(root, text="temp", textvariable=temperature_var)
tk.Label(root, text= "Length:").grid(row=2, sticky="e", padx=5, pady=5)
ln = tk.Entry(root, text="length", textvariable=length_var)

c = tk.Button(root, text="generate", command=generate_text)
output = tk.StringVar(value = storedOut)
o = tk.Label(root, text="", textvariable=output)
b.grid(row=0, column=1, padx=5, pady=5)
t.grid(row=1, column=1, padx=5, pady=5)
ln.grid(row=2, column=1, padx=5, pady=5)
c.grid(row=3, column=0, columnspan=2, pady=10)
o.grid(row=4, column=0, columnspan=2, pady=10)


root.mainloop()

