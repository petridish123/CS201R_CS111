
import sys


def main(input, output_1, output_2):
    input_csv = open(input, "r")
    output_1 = open(output_1, "w")
    output_2 = open(output_2, "w")
    input_lines = input_csv.readlines()
    split = int(len(input_lines)//(5/4))

    out_1 = input_lines[:split]
    out_2 = input_lines[split:]
    print(out_1)
    output_1.writelines(out_1)
    output_2.writelines(out_2)
    

    output_2.close()
    output_1.close()
    input_csv.close()



if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
