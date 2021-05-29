import matplotlib.pyplot as plt
import math





# Learning rate for encoder
def _cosine_lr_generator(step, WARMUP_LR_START, LR_START, LR_FINAL, STEP_WARMUP,STEP_TOTAL):
    # exponential warmup
    if step < STEP_WARMUP:
        warmup_factor = (step / STEP_WARMUP) ** 2
        lr = WARMUP_LR_START + (LR_START - WARMUP_LR_START) * warmup_factor
    # staircase decay
    else:
        factor = math.cos((step - STEP_WARMUP) * 3.14 / (STEP_TOTAL - STEP_WARMUP))
        factor = (factor + 1)/2
        lr =  LR_FINAL * (1 - factor)  + factor * LR_START

    return round(lr, 8)
class CLASS_COSINE_LR():
    def __init__(self, opt, step_total,
                 lr_start, lr_final,
                 warmup_lr_start,warmup_steps
                 ):
        schedule = [_cosine_lr_generator(step, warmup_lr_start, lr_start, lr_final,warmup_steps,step_total) for step in range(step_total)]
        #if DEBUG_ON:
        #plot_lr_schedule(schedule, 'LRPolicy')
        self.opt = opt
        self.schedule = schedule
        self.lr = schedule[0]
        self.opt.learning_rate.assign(self.lr) #initialize lr
        return
    def step(self,step):
        self.lr = self.schedule[step]
        self.opt.learning_rate.assign(self.lr)
        return
    def get_lr(self):
        return self.lr


# plot the learning rate schedule
def plot_lr_schedule(lr_schedule, name):
    plt.figure(figsize=(15,8))
    plt.plot(lr_schedule)
    schedule_info = f'start: {lr_schedule[0]}, max: {max(lr_schedule)}, final: {lr_schedule[-1]}'
    plt.title(f'Step Learning Rate Schedule {name}, {schedule_info}', size=16)
    plt.grid()
    plt.show()