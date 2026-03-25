import React, { useState } from 'react';

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [language, setLanguage] = useState('English');
  const [age, setAge] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleAnalyze = async () => {
    setError(null);
    setResult(null);
    if (!file) {
      setError('Please select a file.');
      return;
    }
    setAnalyzing(true);
    try {
      // 1. Upload file and extract text
      const formData = new FormData();
      formData.append('file', file);
      const uploadRes = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });
      const uploadData = await uploadRes.json();
      if (!uploadRes.ok) throw new Error(uploadData.error || 'Upload failed');

      // 2. Analyze text
      const analyzeForm = new FormData();
      analyzeForm.append('text', uploadData.text);
      analyzeForm.append('language', language);
      analyzeForm.append('age', age);
      const analyzeRes = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: analyzeForm,
      });
      const analyzeData = await analyzeRes.json();
      if (!analyzeRes.ok) throw new Error(analyzeData.error || 'Analyze failed');
      setResult(analyzeData);
    } catch (err: any) {
      setError(err.message || 'Something went wrong');
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: '40px auto', background: '#fff', borderRadius: 16, padding: 32, boxShadow: '0 2px 16px #eee' }}>
      <h2 style={{ marginBottom: 24 }}>Analyze Your Document</h2>
      <input type="file" accept=".pdf,.png,.jpg,.jpeg,.txt" onChange={handleFileChange} />
      <div style={{ margin: '16px 0' }}>
        <label>
          <input type="radio" name="lang" value="English" checked={language === 'English'} onChange={() => setLanguage('English')} /> English
        </label>
        <label style={{ marginLeft: 16 }}>
          <input type="radio" name="lang" value="Hindi" checked={language === 'Hindi'} onChange={() => setLanguage('Hindi')} /> हिंदी
        </label>
      </div>
      <input
        type="text"
        placeholder="Patient Age (Optional)"
        value={age}
        onChange={e => setAge(e.target.value)}
        style={{ width: '100%', marginBottom: 16, padding: 8, borderRadius: 8, border: '1px solid #ccc' }}
      />
      <button
        onClick={handleAnalyze}
        disabled={analyzing}
        style={{ width: '100%', padding: 12, borderRadius: 8, background: '#00b894', color: '#fff', fontWeight: 700, fontSize: 18, border: 'none', cursor: 'pointer' }}
      >
        {analyzing ? 'Analyzing...' : 'Analyze'}
      </button>
      {error && <div style={{ color: 'red', marginTop: 16 }}>{error}</div>}
      {result && (
        <div style={{ marginTop: 32, background: '#f5f7fa', padding: 24, borderRadius: 12 }}>
          <h3>Analysis Result</h3>
          <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}