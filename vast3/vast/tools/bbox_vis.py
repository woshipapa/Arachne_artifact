from PIL import Image

from vast.pipelines import list_pipelines, load_pipeline


def main():
    print(list_pipelines())
    pipe = load_pipeline("detection/grounding_dino/swint_ogc")
    image = Image.open("./1.jpg")
    pred_boxes = pipe(image, "person")[0]
    print(len(pred_boxes))


if __name__ == "__main__":
    main()
