import struct

def main():
    total_size = 3 ** 2
    data = [(j + 1) * 1.0 for j in range(total_size)]
    
    bytes = struct.pack("f" * total_size, *data)

    with open("ch7_input.dat", "wb") as f:
        f.write(bytes)


if __name__ == "__main__":
    main()