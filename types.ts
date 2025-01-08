export interface RequestRecord {
  url: string;
  method: string;
  headers: Record<string, string>;
  resourceType: string;
  postData: any;
  embedding: number[];
}
