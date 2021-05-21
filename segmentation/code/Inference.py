import albumentations as A
from albumentations.pytorch import ToTensorV2

import pandas as pd
from Models import *
from SwinTransformer import *

from Datasets import *



class infer():
    def __init__(self, model, save_path, data_loader,submission_file_name, save_logit ):
        if model == 'deeplabv3_resnet101':
            self.model = Deeplabv3ResNet101()
        elif model == 'deeplabv3_resnet34':
            self.model = Deeplabv3ResNet34()
        elif model == 'deeplabv3_resnext101_32x16d':
            self.model = Deeplabv3ResNext101_32x16d()
        elif model == 'deeplabv3_resnext101_32x4d':
            self.model = Deeplabv3resnext101_32x4d()
        elif model == 'deeplabv3_resnext50_32x4d':
            self.model = Deeplabv3resnext50_32x4d()
        elif model == 'deeplabv3+_resnext50_32x4d':
            self.model = Deeplabv3Plus_resnext50_32x4d()
        elif model == 'deeplabv3+_effinetb0':
            self.model = Deeplabv3Plus_efficientnetb0()
        elif model == 'swin_transformer_base':
            self.model = SwinTransformerBase()
        elif model == 'swin_transformer_small':
            self.model = SwinTransformerSmall()
        elif model == 'swin_transformer_tiny':
            self.model = SwinTransformerTiny()
        else:
            self.model = model

        self.save_path = save_path
        self.data_loader = data_loader
        self.submission_file_name = submission_file_name
        self.save_logit = save_logit

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        checkpoint = torch.load(self.save_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        print("model load", end = '\n\n')

    def test(self):
        size = 256
        transform = A.Compose([A.Resize(256, 256)])
        print('Start prediction.')
        self.model.eval()
        if self.save_logit:
            output_logit = np.empty((1,12,size,size))

        file_name_list = []
        preds_array = np.empty((0, size * size), dtype=np.long)
        batch_size = self.data_loader.batch_size
        print(f"batch_size : {batch_size}")
        print(f"total test data : {len(self.data_loader.dataset)} " )

        total_step = len(self.data_loader)
        with torch.no_grad():
            for step, (imgs, image_infos) in enumerate(self.data_loader):

                # inference (512 x 512)
                outs = self.model(torch.stack(imgs).to(self.device))
                oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
                if self.save_logit:
                    output_logit = np.append(output_logit, outs.detach().cpu().numpy())

                # resize (256 x 256)
                temp_mask = []
                for img, mask in zip(np.stack(imgs), oms):
                    transformed = transform(image=img, mask=mask)
                    mask = transformed['mask']
                    temp_mask.append(mask)

                oms = np.array(temp_mask)

                oms = oms.reshape([oms.shape[0], size * size]).astype(int)
                preds_array = np.vstack((preds_array, oms))

                file_name_list.append([i['file_name'] for i in image_infos])
                print(f"[{step} / {total_step}]\r" , end = "")
        print("End prediction.")
        file_names = [y for x in file_name_list for y in x]
        if self.save_logit:
            np.save('/opt/ml/code/submission/logits/'+self.submission_file_name + '.npy',output_logit[1:])

        return file_names, preds_array


    def submission(self):
        # sample_submisson.csv 열기
        submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)

        # test set에 대한 prediction
        file_names, preds = self.test()

        # PredictionString 대입
        for file_name, string in zip(file_names, preds):
            submission = submission.append(
                {"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())},
                ignore_index=True)

        # submission.csv로 저장
        submission.to_csv("./submission/" + self.submission_file_name + '.csv', index=False)


if __name__ == '__main__':
    def collate_fn(batch):
        return tuple(zip(*batch))

    test_transform = A.Compose([
        # A.CLAHE(p=1),
        ToTensorV2()
    ])
    testdataset = RecylceDatasets(data_dir='../input/data/test.json', mode='test', transform=test_transform,
                                  # resize=512
                                  )
    test_loader = torch.utils.data.DataLoader(dataset=testdataset,
                                              batch_size=8,
                                              num_workers=4,
                                              collate_fn=collate_fn)
    inference = infer(model='swin_transformer_small',
                      save_path='/opt/ml/code/saved/mIoU_swin_Small_allData.pt',
                      data_loader=test_loader,
                      save_logit= False,
                      submission_file_name="24th_swin_Small_512_alldata")
    inference.submission()
