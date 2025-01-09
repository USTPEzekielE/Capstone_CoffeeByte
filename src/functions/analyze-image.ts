import axios from 'axios';
import { Result } from '@/types/result';

export const analyzeImage = async (
  base64Image: string,
): Promise<Result | undefined> => {
  try {
    const response = await axios.post('http://143.198.84.145:5001/predict', {
      image: `data:image/png;base64,${base64Image}`,
    });

    return {
      results: response.data.results,
      image: response.data.image,
    };
  } catch (error) {
    throw error;
  }
};
