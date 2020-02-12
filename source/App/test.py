import tkinter as tk
from App.Utils import DragDropListbox

root = tk.Tk()
list = [1, 2, 3, 4, 5]
listVar = tk.Variable(root, value=list)
listbox = DragDropListbox(root, listvariable=listVar)
listbox.pack(fill=tk.BOTH, expand=True)
root.mainloop()

print(list)
print(listVar.get())