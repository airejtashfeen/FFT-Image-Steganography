import numpy as np
import matplotlib.pyplot as plt
import cv2

# Convert text to binary
def text_to_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)

# Convert bits back to text
def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) == 8:  # Ensure we have a full byte
            chars.append(chr(int(byte, 2)))
    return ''.join(chars)

# Embed text in FFT coefficients
def embed_text_in_fft(fshift, message_bits):
    rows, cols = fshift.shape
    center_row, center_col = rows // 2, cols // 2
    max_bits = 20 * 20  # Embedding region size (20x20 pixels)

    if len(message_bits) > max_bits:
        raise ValueError(f"Message too long! Max bits: {max_bits}, got {len(message_bits)}")

    f_embedded = fshift.copy()
    bit_index = 0

    for i in range(center_row - 10, center_row + 10):
        for j in range(center_col - 10, center_col + 10):
            if bit_index >= len(message_bits):
                return f_embedded

            real_val = f_embedded[i, j].real
            imag_val = f_embedded[i, j].imag

            # Embed bit in the LSB of the integer part of the real value
            int_val = int(np.floor(real_val))
            frac_val = real_val - int_val
            current_bit = int_val & 1
            target_bit = int(message_bits[bit_index])

            if current_bit != target_bit:
                int_val += (target_bit - current_bit)

            f_embedded[i, j] = complex(int_val + frac_val, imag_val)
            bit_index += 1

    return f_embedded

# Extract bits from FFT coefficients
def extract_bits_from_fft(fshift, bit_length):
    rows, cols = fshift.shape
    center_row, center_col = rows // 2, cols // 2

    bits = ''
    for i in range(center_row - 10, center_row + 10):
        for j in range(center_col - 10, center_col + 10):
            if len(bits) >= bit_length:
                return bits

            real_val = fshift[i, j].real
            # Use floor instead of round for robustness, as rounding can flip bits
            int_val = int(np.floor(real_val))
            bits += str(int_val & 1)

    return bits

# === MAIN PIPELINE ===
try:
    # Read image as grayscale
    image = cv2.imread("temp.png")
    if image is None:
        raise FileNotFoundError("Could not load image 'temp.png'. Please check the file path.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    # Prepare message
    secret_msg = "Hello Airej!"
    message_bits = text_to_bits(secret_msg)
    print(f"Original Message: {secret_msg}")
    print(f"Message Bits: {message_bits} ({len(message_bits)} bits)")

    # Embed
    f_embedded = embed_text_in_fft(fshift, message_bits)

    # Inverse FFT
    f_ishift = np.fft.ifftshift(f_embedded)
    img_stego = np.fft.ifft2(f_ishift)
    img_stego_real = np.abs(img_stego)

    # Normalize and convert to uint8 for saving/displaying
    img_stego_display = np.clip(img_stego_real, 0, 255).astype(np.uint8)

    # Save the stego image (optional)
    cv2.imwrite("stego_image.png", img_stego_display)

    # Display image
    plt.imshow(img_stego_display, cmap='gray')
    plt.title("Stego Image")
    plt.axis('off')
    plt.show()

    # Extract message
    f2 = np.fft.fft2(img_stego)  # Use img_stego, not the display version
    f2shift = np.fft.fftshift(f2)
    extracted_bits = extract_bits_from_fft(f2shift, len(message_bits))
    extracted_message = bits_to_text(extracted_bits)

    print("Extracted Bits:", extracted_bits)
    print("Extracted Message:", extracted_message)

except FileNotFoundError as e:
    print(f"Error: {e}")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")