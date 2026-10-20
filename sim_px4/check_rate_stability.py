"""Is PX4's default rate loop stable for a given airframe? Toy rigid-body sweep.

Rerun with measured DD24 inertia / thrust. Run from the repo root with the
Isaac Sim python (needs torch only):  python sim_px4/check_rate_stability.py .
"""
import sys, math, torch
sys.path.insert(0, sys.argv[1])
from environments.px4_rate_emulator import PX4RateEmulator
m=0.82704; g=9.81
for Ixx in (1.5e-3, 3e-3, 5e-3):
  for twr in (2.6, 5.2):
    for tau_m in (0.0, 0.02):
      res=[]
      for hz in (250, 500, 1000):
        dt=1/hz; emu=PX4RateEmulator(1,dt,"cpu"); emu.reset(torch.tensor([0]))
        I=torch.tensor([Ixx,Ixx,1.8*Ixx]); L=0.09/math.sqrt(2); Fmax=twr*m*g/4; thr=1/twr
        px=torch.tensor([1.,-1,1,-1])*L; py=torch.tensor([1.,-1,-1,1])*L; s=torch.tensor([1.,1,-1,-1])
        w=torch.zeros(1,3); Fm=torch.full((4,),thr*Fmax); hist=[]
        for k in range(int(1.0/dt)):
          u=emu.step(torch.tensor([thr]),torch.tensor([[1.0,-1.0,0.5]]),w)
          Fc=u[0]*Fmax; Fm = Fc if tau_m==0 else Fm+(Fc-Fm)*dt/tau_m
          tq=torch.stack([(-py*Fm).sum(),(px*Fm).sum(),(s*0.016*Fm).sum()]); w=w+dt*(tq/I); hist.append(w[0].clone())
        h=torch.stack(hist)[int(0.5/dt):]; pp=(h.max(0).values-h.min(0).values)[:2].max().item(); err=(h[-1]-torch.tensor([1.,-1,0.5])).abs().max().item()
        res.append(f"{hz}Hz pp={pp:.2f} err={err:.2f}")
      print(f"I={Ixx:.1e} TWR={twr} motor_tau={tau_m}: " + " | ".join(res))
