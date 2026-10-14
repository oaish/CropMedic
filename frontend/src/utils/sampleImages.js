export const SAMPLE_PRESETS = [
  {
    id: 'sample-tomato-healthy',
    name: 'Tomato Healthy Leaf',
    filename: 'tomato_healthy_sample.jpg',
    dataUrl: '/samples/tomato_healthy_sample.jpg',
  },
  {
    id: 'sample-tomato-early',
    name: 'Tomato Early Blight',
    filename: 'tomato_early_blight_sample.jpg',
    dataUrl: '/samples/tomato_early_blight_sample.jpg',
  },
  {
    id: 'sample-potato-healthy',
    name: 'Potato Healthy',
    filename: 'potato_healthy_sample.jpg',
    dataUrl: '/samples/potato_healthy_sample.jpg',
  },
  {
    id: 'sample-potato-late',
    name: 'Potato Late Blight',
    filename: 'potato_late_blight_sample.jpg',
    dataUrl: '/samples/potato_late_blight_sample.jpg',
  },
  {
    id: 'sample-pepper-spot',
    name: 'Pepper Bacterial Spot',
    filename: 'pepper_bacterial_spot_sample.jpg',
    dataUrl: '/samples/pepper_bacterial_spot_sample.jpg',
  },
];

export async function dataUrlToFile(dataUrl, filename) {
  const res = await fetch(dataUrl);
  const blob = await res.blob();
  return new File([blob], filename, { type: blob.type });
}