# Lumbar MRI Analyzer v2

این نسخه یک prototype برای localization خودکار disc spaces روی Sagittal T2 است.

## اجرا
```bash
pip install -r requirements.txt
streamlit run app.py
```

## نکته مهم
الگوریتم فعلی medical AI validated نیست. از شدت سیگنال و الگوی افقی تصویر برای پیشنهاد محل دیسک‌ها استفاده می‌کند و ممکن است در تصاویر مختلف اشتباه کند.

## هدف v3
- تشخیص واقعی vertebral bodies با مدل segmentation/detection
- تعیین L1 تا S1 بر اساس anatomy، نه صرفاً موقعیت عمودی
- استفاده از DICOM و کل سری Sagittal T2
- annotation و آموزش مدل MONAI/PyTorch
- تشخیص disc morphology و stenosis بعد از localization
