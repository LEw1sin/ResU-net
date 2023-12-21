from collections import Counter
import torch


class GranularizeTransform:
    def __init__(self, granularity):
        self.granularity = granularity

    def __call__(self, img):
        img = self.granularize_image(img)
        return img

    def compute_weighted_average(self, region):
        number_counts = Counter(region)
        total_counts = sum(number_counts.values())
        weights = {number: count / total_counts for number, count in number_counts.items()}
        weighted_sum = sum(number * weight for number, weight in weights.items())
        return weighted_sum

    def granularize_image(self, image):
        result_image = image.numpy().flatten()
        _, height, width = image.shape

        for i in range(0, result_image.shape[0], self.granularity):
            region = result_image[i:i + self.granularity]
            weighted_sum = self.compute_weighted_average(region)
            result_image[i:i + self.granularity] = weighted_sum

        # result_image.reshape(height, width).unsqueeze(0)
        return torch.from_numpy(result_image.reshape(height, width)).unsqueeze(0)










