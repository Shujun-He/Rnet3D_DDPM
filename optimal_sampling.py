from utils import *
import matplotlib.pyplot as plt
from scipy.stats import norm

config=config=load_config_from_yaml("grid_configs/config_003.yaml")

class diffusion:
    def __init__(self, config):
        beta_1, beta_T = config.beta_min, config.beta_max
        betas = torch.linspace(start=beta_1, end=beta_T, steps=config.n_times)#.to(device) # follows DDPM paper
        self.sqrt_betas = torch.sqrt(betas)
                                     
        # define alpha for forward diffusion kernel
        self.alphas = 1 - betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1-alpha_bars)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)

        self.data_std=config.data_std

model=diffusion(config)

sample_stds=[10, 35, 50, 75, 100]

#set figure size
plt.figure(figsize=(10, 10))

def normal_pdf(x: torch.Tensor,
               mean: float = 0.0,
               std: float  = 1.0) -> torch.Tensor:
    """
    Probability density function of a normal distribution.
    """
    var = std ** 2
    denom = (2 * 3.1415 * var)**0.5
    num   = torch.exp(- (x - mean) ** 2 / (2 * var))
    return num / denom

def get_sample_pdf(std):
    signal_variance_schedule = (model.sqrt_alpha_bars)**2*(std/model.data_std)**2
    center = torch.abs(signal_variance_schedule-0.5).argmin()

    x = torch.linspace(0, 1, config.n_times)
    pdf_vals = normal_pdf(x, mean=center/config.n_times, std=0.2)
    pdf_vals = pdf_vals / pdf_vals.max()  # Normalize the PDF 
    pdf_vals = pdf_vals.float()
    return pdf_vals

for i,std in enumerate(sample_stds):
    # signal_variance_schedule = (model.sqrt_alpha_bars)**2*(std/model.data_std)**2
    # #signal_variance_schedule = signal_variance_schedule/signal_variance_schedule[0]

    # center = torch.abs(signal_variance_schedule-0.5).argmin()

    # x=np.linspace(0, 1, config.n_times)
    # pdf_vals = norm.pdf(x, loc=center/config.n_times, scale=0.25)
    # pdf_vals = torch.tensor(pdf_vals).float()
    pdf_vals = get_sample_pdf(std)
    #exit()
    #print(std, center)
    signal_variance_schedule = (model.sqrt_alpha_bars)**2*(std/model.data_std)**2
    plt.subplot(len(sample_stds), 1, i+1)
    plt.plot(signal_variance_schedule, label=f"std={std}")
    plt.plot(pdf_vals, label=f"pdf std={std}")
    #draw vertical line at center
    #plt.axvline(x=center, linestyle='--', label='Center')
    #plt.axhline(y=0.5, color='g', linestyle='--', label='0.5')

    print(f"std={std}, pdf_max={pdf_vals.max()}, pdf_min={pdf_vals.min()}")
    plt.title("Signal Variance Schedule")
    plt.xlabel("t")
    plt.ylabel("Signal Variance")
    #plt.ylim(0,3)
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.savefig("signal_variance_schedule.png")