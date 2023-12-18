import torch
import numpy as np
import json
import os
import pandas as pd


class AdaptiveDiffusionPipeline:
    def __init__(self, estimator, student, teacher):
        self.estimator = estimator
        self.score_percentiles = None

        self.student = student
        self.teacher = teacher

    def calc_score_percentiles(
        self,
        file_path,
        n_samples,
        num_inference_steps_student,
        prompts_path=None,
    ):
        if os.path.exists(file_path):
            print(f'Loading score percentiles from {file_path}')
            with open(f'{file_path}') as f:
                data = json.load(f)
            self.score_percentiles = {}
            for key in data:
                self.score_percentiles[int(key)] = data[key]
        else:
            print(f'Calculating score percentiles on {n_samples} samples and saving as {file_path}')
            prompts = list(pd.read_csv(prompts_path)['caption'])[:n_samples]
            scores = []
            for prompt in prompts:
                student_out = self.student(prompt=prompt, num_inference_steps=num_inference_steps_student, guidance_scale=0.0).images[0]
                score = self.estimator.score(prompt, student_out)
                scores.append(score)

            score_percentiles = {}
            k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]  # Hard-coded temporary
            for k in k_list:
                score_percentiles[k] = np.percentile(scores, k)

            self.score_percentiles = score_percentiles
            with open(f"{file_path}", "w") as fp:
                json.dump(self.score_percentiles, fp)

    def __call__(
        self,
        prompt,
        num_inference_steps_student=2,
        student_guidance=0.0,
        num_inference_steps_teacher=4,
        teacher_guidance=8.0,
        sigma=0.4,
        k=50,
        seed=0
    ):
        # Step 0. Configuration
        generator = torch.Generator(device="cuda").manual_seed(seed)
        num_all_steps = int(num_inference_steps_teacher / sigma + 1)
        chosen_threshold = self.score_percentiles[k]

        # Step 1. Student prediction
        student_out = self.student(prompt=prompt,
                                   num_inference_steps=num_inference_steps_student,
                                   generator=generator,
                                   guidance_scale=student_guidance).images[0].resize((1024, 1024))

        # Step 2. Score estimation
        reward = self.estimator.score(prompt, student_out)

        # Step 3. Adaptive selection and improvement
        if reward < chosen_threshold:
            final_out = self.teacher(
                prompt=prompt,
                image=student_out,
                num_inference_steps=num_all_steps,
                guidance_scale=teacher_guidance,
                strength=sigma,
            ).images[0]
        else:
            final_out = student_out

        return final_out
