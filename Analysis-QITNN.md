# V1
13/03/26

---

  1. Without stabilization rails, the qutrit geometry barely moves.
  
     Best tests: [Test#1](https://github.com/kaifczxc-lab/qitnn/blob/SiritoriProjects/QITNN-V1-Tests/%231.txt), [Test#2](https://github.com/kaifczxc-lab/qitnn/blob/SiritoriProjects/QITNN-V1-Tests/%232.txt)
     
     Contrast test: [Test#5](https://github.com/kaifczxc-lab/qitnn/blob/SiritoriProjects/QITNN-V1-Tests/%235.txt) or [Test#9](https://github.com/kaifczxc-lab/qitnn/blob/SiritoriProjects/QITNN-V1-Tests/%239.txt)
     
  2. Short context length slows entropy growth and weakens the model.
  
     Best tests: [Test#7](https://github.com/kaifczxc-lab/qitnn/blob/SiritoriProjects/QITNN-V1-Tests/%237.txt), [Test#3](https://github.com/kaifczxc-lab/qitnn/blob/SiritoriProjects/QITNN-V1-Tests/%233.txt)
     
     Long-context reference: [Test#2](https://github.com/kaifczxc-lab/qitnn/blob/SiritoriProjects/QITNN-V1-Tests/%232.txt)
     
  3. The architecture scales down poorly in smaller model sizes.
  
     Best tests: [Test#7](https://github.com/kaifczxc-lab/qitnn/blob/SiritoriProjects/QITNN-V1-Tests/%237.txt), [Test#6](https://github.com/kaifczxc-lab/qitnn/blob/SiritoriProjects/QITNN-V1-Tests/%236.txt)
     
     Intermediate reference: [Test#4](https://github.com/kaifczxc-lab/qitnn/blob/SiritoriProjects/QITNN-V1-Tests/%234.txt)
     
  4. The V/O pathway looks like the most fragile part of the architecture.
  
     Best test: [Test#4](https://github.com/kaifczxc-lab/qitnn/blob/SiritoriProjects/QITNN-V1-Tests/%234.txt)
     
     Secondary support: [Test#5](https://github.com/kaifczxc-lab/qitnn/blob/SiritoriProjects/QITNN-V1-Tests/%235.txt)
     
  5. Better qutrit-state health does not automatically give better cross-entropy.
  
     Best comparison: [Test#5](https://github.com/kaifczxc-lab/qitnn/blob/SiritoriProjects/QITNN-V1-Tests/%235.txt) vs [Test#9](https://github.com/kaifczxc-lab/qitnn/blob/SiritoriProjects/QITNN-V1-Tests/%239.txt)
     
  6. Even in the healthiest regime, the model still does not reach full 3-state utilization.
  
     Best tests: [Test#9](https://github.com/kaifczxc-lab/qitnn/blob/SiritoriProjects/QITNN-V1-Tests/%239.txt), [Test#5](https://github.com/kaifczxc-lab/qitnn/blob/SiritoriProjects/QITNN-V1-Tests/%235.txt)

Verdict: The architecture is probably working, right now its only MVP, thats enough for it
