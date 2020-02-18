import tkinter as tk
from App.Utils import ScrollableFrame

root = tk.Tk()

frame = ScrollableFrame(root)

for i in range(50):
    tk.Button(frame.scrollable_frame, text="Sample scrolling label").pack(fill=tk.BOTH, expand=True)

frame.pack(fill=tk.BOTH, expand=True)
root.mainloop()