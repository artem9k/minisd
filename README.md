Can the dalle-mini treatment be applied to Stable Diffusion?

>100m parameter model can be run on laptops, phone...
YOLOv3 has 33M params. It runs at 4fps on a good phone
A ~100M param model could generate images in near real-time on phones and desktop PCs

### Todo list 
- [ ] Implement DDPM / DDPM-Improved
    - [ ] Train unconditional CIFAR
    - [ ] Train conditional (optional)
- [ ] Implement DDIM
- [ ] Implement LDM
- [ ] Find small conv net architectures that have attention (attention important? can we do 1d conv instead?)
- [ ] RN50 CLIP encoder