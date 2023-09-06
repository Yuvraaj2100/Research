import os
import cv2
import csv

# Define the minimum size threshold for particles
MIN_PARTICLE_SIZE = 100  # Adjust this value as needed
NOISE_THRESHOLD = 1.0  # Adjust this threshold for noise detection

def get_particle_sizes(image):
    # Convert the image to grayscale.
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to find the particles.
    thresholded_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)[1]

    # Find the contours of the particles.
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the sizes of the particles.
    particle_sizes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        particle_sizes.append(area)

    return contours, particle_sizes

def visualize_particles(image, contours, particle_sizes):
    # Draw contours of particles on the image.
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

    # Annotate particle sizes on the image.
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        size = particle_sizes[i]
        cv2.putText(image_with_contours, f"Size: {size:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image_with_contours

def main():
    # Get the path to the folder containing the images.
    image_folder = "/Users/yuvraajbhatter/Desktop/comp_sci/dataset/test"

    # Create a CSV file to store the results.
    csv_file = open("filtered_particle_sizes.csv", "w")
    writer = csv.writer(csv_file)

    # Write a header row to the CSV file with dynamically generated column names
    header = ["Image"]
    for i in range(1, 11):  # Assuming a maximum of 10 particles per image
        header.append(f"Particle {i} Size")
    writer.writerow(header)

    # Iterate over the images in the folder.
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)

        # Check if the file exists
        if not os.path.isfile(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        # Read the image.
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Failed to load image: {image_path}")
                continue
        except Exception as e:
            print(f"Error while loading image {image_path}: {str(e)}")
            continue

        # Get the particle sizes and contours.
        contours, particle_sizes = get_particle_sizes(image)

        # Filter out particles based on size (remove noise)
        filtered_particle_sizes = [size for size in particle_sizes if size >= MIN_PARTICLE_SIZE]

        # Check if there are any valid particle sizes (not noise) to write
        if any(size > NOISE_THRESHOLD for size in filtered_particle_sizes):
            # Create a list to store particle sizes for this image
            particle_size_list = [filename]

            # Append the filtered particle sizes to the list (up to a maximum of 10)
            particle_count = 0
            for size in filtered_particle_sizes:
                if size > NOISE_THRESHOLD:
                    particle_count += 1
                    particle_size_list.append(size)
                if particle_count >= 10:
                    break

            # Fill remaining columns with empty strings if necessary
            while len(particle_size_list) < 11:
                particle_size_list.append("")

            # Write the particle sizes to the CSV file
            writer.writerow(particle_size_list)

    csv_file.close()

if __name__ == "__main__":
    main()
