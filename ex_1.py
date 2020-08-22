## การเปิดและอ่านไฟล์ npz ที่เป็นไฟล์ข้อมูลของโมเลกุล Benzene

import numpy as np
# https://octadist.github.io/
import octadist as oc

# http://www.quantum-machine.org/gdml/data/npz/benzene_dft.npz
data = np.load('dataset/benzene_dft.npz')
print(data.files)
print(data['name']) # name
print(data['z']) # atomic number
print(data['R']) # atomic symbol
print(data['E']) # energies
print(data['F']) # forces

# ---------------

atomic_number = data['z'].tolist()
print(atomic_number)

atomic_symbol = [oc.check_atom(x) for x in atomic_number] # <<<<<< atomic symbols
print(atomic_symbol)

coor_1 = data['R'][0] # <<<<< coord of conformer 1

# ---------------

mol = oc.DrawComplex(atom=atomic_symbol, coord=coor_1)
mol.add_atom()
mol.add_bond()
mol.add_legend()
mol.add_symbol()
# mol.save_img()
mol.show_plot()
