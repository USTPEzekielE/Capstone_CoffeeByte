import { TypedArray } from '@/types';

export function getPrediction(response: TypedArray[]) {
  const classes = ['Cercospora', 'Coffee Rust', 'Healthy', 'Phoma'];

  const array = response[0];
  let maxValue = Number(array[0]);
  let index = 0;

  for (let i = 1; i < array.length; i++) {
    const value =
      typeof array[i] === 'bigint' ? Number(array[i]) : (array[i] as number);

    if (value > maxValue) {
      maxValue = value;
      index = i;
    }
  }

  console.log(
    'index',
    index,
    'adagradResult: ',
    response,
    'maxValue: ',
    maxValue,
  );

  return {
    class: classes[index],
    confidence: array[index],
    index: index,
  };
}
