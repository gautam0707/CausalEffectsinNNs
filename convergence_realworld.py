import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'size': 12}
matplotlib.rc('font', **font)

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(nrows=120, ncols=320)
ax1 = fig.add_subplot(gs[10:50, 0:100])
ax2 = fig.add_subplot(gs[10:50, 110:210])
ax3 = fig.add_subplot(gs[10:50, 220:320])

line_width = 2
ensemble_size = 5

# Auto-MPG
p1_losses = []
p2_losses = []
for ensemble in range(ensemble_size):
    f_l = np.load('./autompg/losses/ahce_autompg_freeze_'+str(ensemble+1)+'.npy')
    uf_l = np.load('./autompg/losses/ahce_autompg_unfreeze_'+str(ensemble+1)+'.npy')
    p1_losses.append(f_l)
    p2_losses.append(uf_l)
p1_losses = np.array(p1_losses)/322
p2_losses = np.array(p2_losses)/322

ax1.plot(np.mean(p1_losses, axis=0), label='Phase 1', linewidth=line_width)
ax1.plot(np.mean(p2_losses, axis=0), label='Phase 2', linewidth=line_width)
ax1.set_ylabel('RMSE')
ax1.legend(frameon=False)
ax1.set_title('Auto-MPG')
ax1.set_xlabel('Epochs')

# Lung Cancer
p1_losses = []
p2_losses = []
for ensemble in range(ensemble_size-1):
    f_l = np.load('./lungcancer/losses/ahce_lungcance_freeze_'+str(ensemble+1)+'.npy')
    uf_l = np.load('./lungcancer/losses/ahce_lungcancer_unfreeze_'+str(ensemble+1)+'.npy')
    p1_losses.append(f_l)
    p2_losses.append(uf_l)
p1_losses = np.array(p1_losses)
p2_losses = np.array(p2_losses)

ax2.plot(np.mean(p1_losses, axis=0), label='Phase 1', linewidth=line_width)
ax2.plot(np.mean(p2_losses, axis=0), label='Phase 2', linewidth=line_width)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend(frameon=False)
ax2.set_title('Lung Cancer')

# Sachs
p1_losses = []
p2_losses = []
for ensemble in range(2):
    f_l = np.load('./sachs/losses/ahce_sachs_freeze_'+str(ensemble+1)+'.npy')
    uf_l = np.load('./sachs/losses/ahce_sachs_unfreeze_'+str(ensemble+1)+'.npy')
    p1_losses.append(f_l)
    p2_losses.append(uf_l)
p1_losses = np.array(p1_losses)
p2_losses = np.array(p2_losses)

ax3.plot(np.mean(p1_losses, axis=0), label='Phase 1', linewidth=line_width)
ax3.plot(np.mean(p2_losses, axis=0), label='Phase 2', linewidth=line_width)
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Accuracy')
ax3.legend(frameon=False)
ax3.set_title('Sachs')

plt.savefig('convergence.png')
plt.savefig('convergence.svg', format='svg',dpi=1000)