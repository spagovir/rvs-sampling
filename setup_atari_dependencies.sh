pip install atari-py
mkdir atari-roms
cd atari-roms
curl http://www.atarimania.com/roms/Roms.rar -O Roms.rar
sudo apt install unrar
unrar x Roms.rar
cd ..
python -m atari_py.import_roms atari-roms/ROMS/
pip install ale-py
ale-import-roms atari-roms/ROMS/
