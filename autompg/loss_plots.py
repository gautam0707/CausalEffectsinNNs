import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pdb

font = {'size'   : 22}
matplotlib.rc('font', **font)

ensemble_size = 5
epochs = 30
p1_losses = []
p2_losses = []
for ensemble in range(ensemble_size):
    f_l = np.load('./losses/ahce_autompg_freeze_'+str(ensemble+1)+'.npy')
    uf_l = np.load('./losses/ahce_autompg_unfreeze_'+str(ensemble+1)+'.npy')
    p1_losses.append(f_l)
    p2_losses.append(uf_l)
p1_losses = np.array(p1_losses)/322
p2_losses = np.array(p2_losses)/322

# pdb.set_trace()
plt.title('Auto-MPG')
plt.plot(np.mean(p1_losses, axis=0), label='Phase 1', linewidth=3)
plt.plot(np.mean(p2_losses, axis=0), label='Phase 2', linewidth=3)
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.tight_layout()
plt.savefig('autompg_loss.png')
