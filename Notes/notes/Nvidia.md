# Nvidia

In this part, I will talk about some nvidia commands. Of course, you can find more on NVIDIA official website. 

## nvidia-smi

Which means NVIDIA System Management Interface. 

We can get information about our GPU cards using this command, include but not limited to the number of cards, driver version, CUDA version, memory usage, volatile GPU-util, etc. 

See for yourself ~ 

### nvidia-smi --help/-h

For more information about how to use `nvidia-smi` or we can say that <strong>open a gate to new world.</strong> 

### nvidia-smi --list-gpus/-L

Display all GPUs connected to your system. 

### nvidia-smi --list-blacklist-gpus/-B

Display all blacklisted GPUs. 

### nvidia-smi --id/-i (number)

Target a sprcific GPU. 


... and so on ... maybe update if I'm not occupied. 

## Device Monitoring

### nvidia-smi dmon

Display all devices stats in scrolling format. Use `-i` to target a specific GPU. 

This command is really helpful. For example, 

I train a VGG16 and I want to raise my gpu-util. But first, I need to konw my gpu-util during training. `nvidia-smi dmon` will work. 
