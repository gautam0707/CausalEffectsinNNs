import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'size': 14}
matplotlib.rc('font', **font)

fig = plt.figure(constrained_layout=False)
gs = fig.add_gridspec(nrows=120, ncols=110)
ax1 = fig.add_subplot(gs[10:50, 10:50])
ax2 = fig.add_subplot(gs[10:50, 70:110])
ax3 = fig.add_subplot(gs[70:110, 10:50])
ax4 = fig.add_subplot(gs[70:110, 70:110])

line_width = 2
ensemble_size = 5
# Synthetic Data
p1_losses = []
p2_losses = []
for ensemble in range(ensemble_size):
    f_l = np.load('./synthetic/losses/ahce_s1_freeze_'+str(ensemble+1)+'.npy')
    uf_l = np.load('./synthetic/losses/ahce_s1_unfreeze_'+str(ensemble+1)+'.npy')
    p1_losses.append(f_l)
    p2_losses.append(uf_l)
p1_losses = np.array(p1_losses)/9950
p2_losses = np.array(p2_losses)/9950
print(p1_losses)
ax1.plot(np.mean(p1_losses, axis=0), label='Phase 1', linewidth=line_width)
ax1.plot(np.mean(p2_losses, axis=0), label='Phase 2', linewidth=line_width)
ax1.set_ylabel('RMSE')
ax1.legend(frameon=False)
ax1.set_title('Synthetic Data')


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

ax2.plot(np.mean(p1_losses, axis=0), label='Phase 1', linewidth=line_width)
ax2.plot(np.mean(p2_losses, axis=0), label='Phase 2', linewidth=line_width)
ax2.set_ylabel('RMSE')
ax2.legend(frameon=False)
ax2.set_title('Auto-MPG')


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

ax3.plot(np.mean(p1_losses, axis=0), label='Phase 1', linewidth=line_width)
ax3.plot(np.mean(p2_losses, axis=0), label='Phase 2', linewidth=line_width)
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Accuracy')
ax3.legend(frameon=False)
ax3.set_title('Lung Cancer')

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

ax4.plot(np.mean(p1_losses, axis=0), label='Phase 1', linewidth=line_width)
ax4.plot(np.mean(p2_losses, axis=0), label='Phase 2', linewidth=line_width)
ax4.set_xlabel('Epochs')
ax4.set_ylabel('Accuracy')
ax4.legend(frameon=False)
ax4.set_title('Sachs')

plt.savefig('convergence.png')
plt.savefig('convergence.svg', format='svg',dpi=1000)