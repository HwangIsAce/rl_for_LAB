import imageio

# imaage to gif 
def save_gif(images, filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for image in images:
            writer.append_data(image)