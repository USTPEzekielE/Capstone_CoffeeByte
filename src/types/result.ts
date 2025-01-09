export interface Result {
  image: string;
  results: {
    class: number;
    confidence: number;
    bbox: number[];
  }[];
}
