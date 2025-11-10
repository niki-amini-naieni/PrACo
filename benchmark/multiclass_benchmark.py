from .benchmark import Benchmark


class MulticlassBenchmark(Benchmark):
    def __init__(self, model, img_class_txt_path, benchmark_results_dir="multiclass_benchmark_results", img_class_dict=None):
        super().__init__(model, img_class_txt_path, benchmark_results_dir, img_class_dict)

    def run_negative_label_test(self, output_csv, split="test", force=False):

        return super().run_negative_label_test(output_csv, split, force)
    
    def run_mosaic_test(self, img_classes, output_upper_csv, output_lower_csv, split="test", force=False):
        
        raise NotImplementedError("Mosaic test not needed in MulticlassBenchmark.")