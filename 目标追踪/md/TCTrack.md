# TCTrack

### create model

- model = ModelBuilder_tctrack('train').train()
  - self.backbone = TemporalAlexNet().cuda()
    - backbone的构成是AlexNet的变形，将5个卷积层的后两个替换为论文中提到的TAdaConv