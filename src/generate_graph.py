import argparse
import matplotlib.pyplot as plt

def gen_graph(loss_path, output_path):
    disc_loss = []
    gen_loss = []
    with open(loss_path, 'r') as f:
        for line in f.readlines():
            s = line.split()
            gen_loss.append(float(s[3]))
            disc_loss.append(float(s[6]))
    plt.title('Implemented Method')
    gen_plot = plt.plot(gen_loss, label='Generator Loss')
    disc_plot = plt.plot(disc_loss, label='Discriminator Loss')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-path',
                        type=str,
                        required=True)
    parser.add_argument('--output-path',
                        type=str,
                        default='')
    args = parser.parse_args()
    loss_path = args.loss_path
    output_path = args.output_path
    gen_graph(loss_path, output_path)