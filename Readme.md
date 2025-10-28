# Creazione di un ambiente virtuale python
```bash
python3 -m venv venv
source venv/bin/activate
```

# Installazione di pyTorch
Segui le istruzioni della [guida ufficiale di pytorch](https://pytorch.org/get-started/locally/). 

Su Linux, con Cuda 13 (dall'ambiente virtuale)
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

# Installazione di snnTorch e altri pacchetti necessari

```bash
pip3 install -r requirements.txt
```