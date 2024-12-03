import { TypedArray } from '@/types';

interface ScanResult {
  type: 'normal' | 'disease' | 'biotic';
  confidence: number;
  details?: string;
}

export const determineResult = (
  resnetResults: TypedArray[] | null,
  diseaseResults: TypedArray[] | null,
  bioticResults: TypedArray[] | null,
): ScanResult => {
  // Convert TypedArray to regular array and get maximum value
  const getMaxConfidence = (response: TypedArray[] | null): number => {
    if (!response || !response[0]) {
      return 0;
    }

    const array = response[0];
    let maxValue = Number(array[0]); // Initialize with first value

    // Loop through array to find maximum value
    for (let i = 1; i < array.length; i++) {
      const value =
        typeof array[i] === 'bigint' ? Number(array[i]) : (array[i] as number);
      if (value > maxValue) {
        maxValue = value;
      }
    }
    return maxValue;
  };

  const resnetConfidence = getMaxConfidence(resnetResults);
  const diseaseConfidence = getMaxConfidence(diseaseResults);
  const bioticConfidence = getMaxConfidence(bioticResults);

  console.log('Confidence scores:', {
    resnet: resnetConfidence,
    disease: diseaseConfidence,
    biotic: bioticConfidence,
  });

  // You can adjust these threshold values based on your needs
  const CONFIDENCE_THRESHOLD = 0.6;

  if (
    resnetConfidence > CONFIDENCE_THRESHOLD &&
    resnetConfidence > diseaseConfidence &&
    resnetConfidence > bioticConfidence
  ) {
    return {
      type: 'normal',
      confidence: resnetConfidence,
      details: 'Normal plant detected',
    };
  } else if (
    diseaseConfidence > CONFIDENCE_THRESHOLD &&
    diseaseConfidence > bioticConfidence
  ) {
    return {
      type: 'disease',
      confidence: diseaseConfidence,
      details: 'Disease detected',
    };
  } else if (bioticConfidence > CONFIDENCE_THRESHOLD) {
    return {
      type: 'biotic',
      confidence: bioticConfidence,
      details: 'Biotic stress detected',
    };
  } else {
    // Default case when confidence is too low
    return {
      type: 'normal',
      confidence: Math.max(
        resnetConfidence,
        diseaseConfidence,
        bioticConfidence,
      ),
      details: 'Unable to determine with high confidence',
    };
  }
};
