import json
import pprint


def main():
    with open("data/test.jsonl") as f:
        for line in f:
            data = json.loads(line)
            pprint.pprint(data)
            break



if __name__ == '__main__':
    main()