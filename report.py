"""Build the 3-page comparison report from the saved run results."""
import json
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent


class Report(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Times", "I", 9)
        self.set_text_color(80)
        self.cell(0, 6, f"{self.page_no()} / 3", align="C")
        self.set_text_color(0)

    def heading(self, text):
        self.set_font("Times", "B", 13)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def body(self, text):
        self.set_font("Times", "", 11)
        self.multi_cell(0, 5, text)
        self.ln(1.5)


def main() -> None:
    pytorch = json.loads((ROOT / "results/pytorch/metrics.json").read_text())
    roboflow = json.loads((ROOT / "results/roboflow/summary.json").read_text())
    train_min = pytorch["train_seconds"] / 60

    pdf = Report(format="Letter")
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    pdf.set_font("Times", "B", 16)
    pdf.multi_cell(0, 7, "Fabric defect detection: a Roboflow and PyTorch comparison")
    pdf.ln(1)
    pdf.set_font("Times", "I", 11)
    pdf.multi_cell(
        0,
        5,
        "Single-label classification of fabric images into Defect free, Defected, and Waste. "
        "Both models are ResNet-18, trained on the same 70/20/10 split.",
    )
    pdf.ln(2)

    pdf.heading("1. Dataset")
    pdf.body(
        "The images were labeled in Roboflow and frozen as dataset version 1 "
        "(Poorva Fabric defect detection, IDM 2), exported in folder format on 29 October 2026. "
        "The export contains 4,995 source images: 3,499 train, 999 validation, and 497 test, "
        "which is the requested 70/20/10 split. Every image was auto-oriented and stretched to "
        "224x224. Two augmentations were applied to the training images only: a horizontal flip "
        "with probability 50%, and a brightness change between -15% and +15%. Roboflow kept two "
        "copies of each training image, so the training folder has 6,998 files."
    )
    pdf.body(
        "The classes are uneven. The training folder holds 6,304 Defected, 670 Defect free, and "
        "24 Waste images. The test folder holds 449 Defected, 47 Defect free, and 1 Waste image. "
        "A model that almost always predicts Defected will score well on overall accuracy. "
        "Per-class recall is the number that shows whether Defect free and Waste are actually recognized."
    )

    pdf.heading("2. The two models")
    pdf.body(
        "The Roboflow model is "
        "sanya-mohanty/poorva-fabric-defect-detection-idm-2-1-resnet18-t1. "
        "It is a ResNet-18 started from the ImageNet checkpoint, trained with Roboflow's default "
        "hyperparameters. Inference in this benchmark calls the hosted API at serverless.roboflow.com. "
        "On the test split, Roboflow's evaluation reports precision 94.5%, recall 91.7%, and F1 93.0% "
        "at an operating threshold of 61% confidence."
    )
    pdf.body(
        f"The PyTorch model is also a ResNet-18, loaded from ImageNet weights with the final layer "
        f"replaced by three outputs, then fine-tuned on the exported train and validation folders. "
        f"Training used AdamW with learning rate 0.0003, weight decay 0.0001, batch size 32, and a "
        f"cosine learning-rate schedule for 15 epochs. The training loader applied the same two "
        f"augmentations, random horizontal flip and brightness jitter of 15%. Validation and test "
        f"images were only resized and normalized. The run used the Apple GPU (MPS), took "
        f"{train_min:.1f} minutes ({pytorch['train_seconds']:.0f} seconds), and reached its best "
        f"validation accuracy of {pytorch['best_valid_acc']:.1%} at epoch {pytorch['best_epoch']}."
    )
    pdf.body(
        "Both models are scored on one shared list of 50 test images, saved in eval_images.txt so "
        "the comparison cannot drift. The list has 25 Defect free, 24 Defected, and 1 Waste image. "
        "It contains every Waste test image and about half of the Defect free test images, so it is "
        "a harder sample than the full test set. predictions.csv holds the Roboflow output for these "
        "50 images: file name, predicted label, and confidence."
    )

    pdf.add_page()
    pdf.heading("3. Accuracy, training time, and latency")
    pdf.set_font("Times", "", 10)
    with pdf.table(
        width=176,
        col_widths=(72, 54, 54),
        text_align=("LEFT", "CENTER", "CENTER"),
        line_height=5,
    ) as table:
        header = table.row()
        header.cell("Measure", align="L")
        header.cell("Roboflow ResNet-18")
        header.cell("PyTorch ResNet-18")
        rows = [
            ("Test precision", "94.5%", "92.7% macro"),
            ("Test recall", "91.7%", "90.2% macro"),
            ("Test F1", "93.0%", "91.4% macro"),
            ("Test accuracy", "see note below", "95.8% (476/497)"),
            ("Accuracy on the same 50 images", "82% (41/50)", "78% (39/50)"),
            ("Training time", "from the training email", f"{train_min:.1f} min"),
            ("Mean latency on the 50", f"{roboflow['latency_ms_mean']:.0f} ms", f"{pytorch['eval_50']['latency_ms_mean']:.1f} ms"),
            ("Median latency on the 50", f"{roboflow['latency_ms_median']:.0f} ms", f"{pytorch['eval_50']['latency_ms_median']:.1f} ms"),
        ]
        for row in rows:
            line = table.row()
            for value in row:
                line.cell(value)
    pdf.ln(2)
    pdf.body(
        "Read the two accuracy columns on the same footing. Roboflow's 94.5 / 91.7 / 93.0 figures "
        "come from its model evaluation on the test split, taken at the 61% threshold. PyTorch has "
        "no reject option: every image gets the class with the highest score, and the precision, "
        "recall, and F1 above are macro averages, so each class counts equally. Under that rule the "
        "PyTorch model scores 95.8% accuracy on all 497 test images. Most of that score is the "
        "Defected class, which it recalls at 98.2% (441 of 449). Defect free recall is 72.3% "
        "(34 of 47). The single Waste test image is correct, and one image is not evidence."
    )
    pdf.body(
        "On the shared 50 images, Roboflow is correct on 41 and PyTorch on 39. The four-point gap "
        "is two images. Both percentages sit well below the full-test accuracy because this sample "
        "is half Defect free, and Defect free is the class both models miss."
    )
    pdf.body(
        f"Latency is where the two setups differ. Roboflow's mean of {roboflow['latency_ms_mean']:.0f} ms "
        f"per image (median {roboflow['latency_ms_median']:.0f} ms, 95th percentile "
        f"{roboflow['latency_ms_p95']:.0f} ms) includes sending the file to the hosted API and waiting "
        f"for the response. PyTorch's mean of {pytorch['eval_50']['latency_ms_mean']:.1f} ms is measured "
        f"on the same machine and includes loading the image and preprocessing; the forward pass alone "
        f"averages {pytorch['eval_50']['forward_ms_mean']:.1f} ms. The hosted number is the cost of "
        f"calling the API. The local number is the cost of running the network beside the camera."
    )
    pdf.body(
        f"PyTorch training took {train_min:.1f} minutes for 15 epochs, about 77 seconds per epoch, "
        "on a laptop GPU. Roboflow does not write its training duration into the exported metrics. "
        "The duration is on the training job, or in the completion email, and that value should be "
        "entered in the table above before the report is submitted. The hosted job used Roboflow's "
        "default recipe. The local job used the learning rate, batch size, and epoch count in section 2."
    )

    pdf.add_page()
    pdf.heading("4. Failure analysis")
    pdf.body(
        "Of the 50 shared images, 11 are wrong in at least one model, and every one of them is a "
        "Defect free image predicted as Defected. Nine are wrong in both models, most of them with "
        "confidence above 0.97. Two are wrong only in PyTorch. On defect-free-137, Roboflow scores "
        "Defect free at 0.616 and PyTorch scores Defected at 0.899. On defect-free-54, Roboflow "
        "scores Defect free at 0.638 and PyTorch scores Defected at 0.613. Those two Roboflow "
        "decisions sit just above the 61% threshold, so they are shallow wins."
    )
    pdf.body(
        "The full PyTorch test set shows the same boundary. It makes 21 errors: 13 Defect free "
        "images called Defected, and 8 Defected images called Defect free. Waste is never confused "
        "with either class. The PyTorch confusion matrix is:"
    )
    pdf.set_font("Times", "", 10)
    with pdf.table(width=150, col_widths=(42, 36, 36, 36), text_align="CENTER", line_height=5) as table:
        header = table.row()
        for text in ("True \\ predicted", "Defect free", "Defected", "Waste"):
            header.cell(text)
        for label, counts in (
            ("Defect free", ["34", "13", "0"]),
            ("Defected", ["8", "441", "0"]),
            ("Waste", ["0", "0", "1"]),
        ):
            line = table.row()
            line.cell(label)
            for count in counts:
                line.cell(count)
    pdf.ln(2)
    pdf.body(
        "The missed images are hard to tell apart by eye once they have been stretched to 224x224 "
        "and compressed. Many of the files are only 2 to 5 KB. A plain measure of texture, the "
        "standard deviation of the grayscale pixels, lines up with the mistakes. A typical Defect "
        "free test image has a standard deviation near 3. The Defected images called Defect free "
        "are the smoothest Defected images, down to 2.2. The Defect free images called Defected "
        "include the most textured Defect free images, with a standard deviation near 16, close to "
        "the Defected median of 11.8. Both networks are separating swatches by how much the weave "
        "varies, and on these edge cases that variation does not match the label. Because the wrong "
        "answer often comes with confidence above 0.99, raising the threshold does not remove them. "
        "The Waste test image is a fold rather than a stain, and both models label it Waste. With "
        "12 Waste images in the original training split and one in the test split, that single "
        "correct call is not a result to lean on."
    )
    pdf.heading("5. Conclusion")
    pdf.body(
        "The two ResNet-18 trainings land in the same place on accuracy. Roboflow's published test "
        "F1 is 93.0%, and the PyTorch macro F1 is 91.4%. On the 50 shared images Roboflow is two "
        "images ahead, 82% against 78%. The local model answers in about 11 ms, against about 382 ms "
        "for the hosted API, and it trained in 19.2 minutes on a laptop. The errors are concentrated "
        "on one boundary: smooth Defected swatches and textured Defect free swatches. More images "
        "in Defect free and Waste, and a pass over the labels where those two classes look alike, "
        "would change the result more than a larger network would."
    )

    out = ROOT / "report.pdf"
    pdf.output(out)
    print(f"Wrote {out} ({pdf.pages_count} pages)")


if __name__ == "__main__":
    main()
