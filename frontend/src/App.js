import React, { useState, useRef } from 'react';
import { Upload, Camera, Sparkles, Heart, ShoppingBag, Loader2, Check, X } from 'lucide-react';

const FashionRecommendationApp = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const API_BASE_URL = 'http://localhost:8000'; // Change this to your API URL

  const handleImageSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setResults(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedImage);
    formData.append('num_recommendations', '3');

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(`Failed to get recommendations: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const resetUpload = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    setResults(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const RecommendationCard = ({ item, category }) => (
    <div style={styles.recommendationCard}>
      <div style={styles.imageContainer}>
        {item.image_url ? (
          <img
            src={item.image_url}
            alt={`${category} item`}
            style={styles.itemImage}
            onError={(e) => {
              e.target.style.display = 'none';
              e.target.nextSibling.style.display = 'flex';
            }}
          />
        ) : null}
        <div style={styles.placeholderImage}>
          <Camera size={48} />
        </div>
      </div>
      <div style={styles.cardContent}>
        <h4 style={styles.cardTitle}>{item.productDisplayName || 'Fashion Item'}</h4>
        <div style={styles.cardDetails}>
          <p><span style={styles.detailLabel}>Color:</span> {item.baseColour}</p>
          <p><span style={styles.detailLabel}>Season:</span> {item.season}</p>
          <p><span style={styles.detailLabel}>Usage:</span> {item.usage}</p>
        </div>
      </div>
    </div>
  );

  const styles = {
    container: {
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #faf5ff 0%, #fef7f7 50%, #f0f9ff 100%)',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    },
    header: {
      background: 'rgba(255, 255, 255, 0.8)',
      backdropFilter: 'blur(10px)',
      borderBottom: '1px solid rgba(229, 231, 235, 0.5)',
      position: 'sticky',
      top: 0,
      zIndex: 50
    },
    headerContent: {
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '16px',
      display: 'flex',
      alignItems: 'center',
      gap: '12px'
    },
    headerIcon: {
      padding: '8px',
      background: 'linear-gradient(135deg, #a855f7, #ec4899)',
      borderRadius: '12px'
    },
    headerTitle: {
      fontSize: '24px',
      fontWeight: 'bold',
      background: 'linear-gradient(to right, #9333ea, #ec4899)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
      backgroundClip: 'text'
    },
    mainContent: {
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '32px 16px'
    },
    uploadSection: {
      marginBottom: '48px'
    },
    sectionTitle: {
      textAlign: 'center',
      marginBottom: '32px'
    },
    mainTitle: {
      fontSize: '30px',
      fontWeight: 'bold',
      color: '#1f2937',
      marginBottom: '16px'
    },
    subtitle: {
      color: '#6b7280',
      maxWidth: '600px',
      margin: '0 auto',
      lineHeight: 1.6
    },
    uploadContainer: {
      maxWidth: '600px',
      margin: '0 auto'
    },
    uploadArea: {
      border: '2px dashed #c084fc',
      borderRadius: '16px',
      padding: '48px',
      textAlign: 'center',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      ':hover': {
        borderColor: '#a855f7',
        backgroundColor: 'rgba(196, 132, 252, 0.1)'
      }
    },
    uploadIcon: {
      padding: '16px',
      background: 'linear-gradient(135deg, #a855f7, #ec4899)',
      borderRadius: '50%',
      marginBottom: '16px',
      display: 'inline-block'
    },
    uploadText: {
      fontSize: '18px',
      fontWeight: '500',
      color: '#374151',
      marginBottom: '8px'
    },
    uploadSubtext: {
      fontSize: '14px',
      color: '#6b7280'
    },
    previewContainer: {
      background: 'white',
      borderRadius: '16px',
      boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1)',
      overflow: 'hidden'
    },
    previewImage: {
      width: '100%',
      aspectRatio: '16/9',
      background: '#f3f4f6',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center'
    },
    previewImg: {
      maxWidth: '100%',
      maxHeight: '100%',
      objectFit: 'contain'
    },
    buttonContainer: {
      padding: '24px',
      display: 'flex',
      gap: '12px'
    },
    primaryButton: {
      flex: 1,
      background: 'linear-gradient(to right, #a855f7, #ec4899)',
      color: 'white',
      padding: '12px 24px',
      borderRadius: '12px',
      border: 'none',
      fontWeight: '500',
      cursor: 'pointer',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '8px',
      transition: 'all 0.3s ease',
      fontSize: '16px'
    },
    secondaryButton: {
      padding: '12px 24px',
      border: '1px solid #d1d5db',
      color: '#374151',
      borderRadius: '12px',
      background: 'white',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      fontSize: '16px'
    },
    hiddenInput: {
      display: 'none'
    },
    errorContainer: {
      maxWidth: '600px',
      margin: '0 auto 32px'
    },
    errorBox: {
      background: '#fef2f2',
      border: '1px solid #fecaca',
      borderRadius: '12px',
      padding: '16px',
      display: 'flex',
      alignItems: 'center',
      gap: '12px'
    },
    errorText: {
      color: '#dc2626'
    },
    resultsSection: {
      display: 'flex',
      flexDirection: 'column',
      gap: '48px'
    },
    detectedSection: {
      background: 'white',
      borderRadius: '16px',
      boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1)',
      padding: '32px'
    },
    sectionHeader: {
      display: 'flex',
      alignItems: 'center',
      gap: '12px',
      marginBottom: '24px'
    },
    sectionHeaderTitle: {
      fontSize: '24px',
      fontWeight: 'bold',
      color: '#1f2937'
    },
    metadataGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
      gap: '16px'
    },
    metadataItem: {
      background: '#f9fafb',
      borderRadius: '12px',
      padding: '16px',
      textAlign: 'center'
    },
    metadataLabel: {
      fontSize: '14px',
      color: '#6b7280',
      marginBottom: '4px',
      textTransform: 'capitalize'
    },
    metadataValue: {
      fontWeight: '600',
      color: '#1f2937'
    },
    recommendationsContainer: {
      display: 'flex',
      flexDirection: 'column',
      gap: '32px'
    },
    categoryTitle: {
      fontSize: '20px',
      fontWeight: '600',
      color: '#1f2937',
      marginBottom: '16px',
      textTransform: 'capitalize'
    },
    recommendationsGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
      gap: '24px'
    },
    recommendationCard: {
      background: 'white',
      borderRadius: '12px',
      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
      overflow: 'hidden',
      transform: 'scale(1)',
      transition: 'all 0.3s ease',
      ':hover': {
        transform: 'scale(1.05)',
        boxShadow: '0 8px 25px rgba(0, 0, 0, 0.15)'
      }
    },
    imageContainer: {
      position: 'relative',
      aspectRatio: '1',
      background: 'linear-gradient(135deg, #f3f4f6, #e5e7eb)'
    },
    itemImage: {
      width: '100%',
      height: '100%',
      objectFit: 'cover'
    },
    placeholderImage: {
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#9ca3af'
    },
    cardContent: {
      padding: '16px'
    },
    cardTitle: {
      fontWeight: '600',
      color: '#1f2937',
      marginBottom: '8px',
      fontSize: '16px'
    },
    cardDetails: {
      fontSize: '14px',
      color: '#6b7280',
      display: 'flex',
      flexDirection: 'column',
      gap: '4px'
    },
    detailLabel: {
      fontWeight: '500'
    },
    noResults: {
      background: '#fffbeb',
      border: '1px solid #fed7aa',
      borderRadius: '12px',
      padding: '24px',
      textAlign: 'center'
    },
    noResultsIcon: {
      color: '#f59e0b',
      margin: '0 auto 16px'
    },
    noResultsText: {
      color: '#92400e'
    },
    footer: {
      marginTop: '64px',
      borderTop: '1px solid rgba(229, 231, 235, 0.5)',
      background: 'rgba(255, 255, 255, 0.5)',
      backdropFilter: 'blur(10px)'
    },
    footerContent: {
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '32px 16px',
      textAlign: 'center',
      color: '#6b7280'
    }
  };

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <div style={styles.headerContent}>
          <div style={styles.headerIcon}>
            <Sparkles size={24} color="white" />
          </div>
          <h1 style={styles.headerTitle}>
            DressCode-AI
          </h1>
        </div>
      </div>

      <div style={styles.mainContent}>
        {/* Upload Section */}
        <div style={styles.uploadSection}>
          <div style={styles.sectionTitle}>
            <h2 style={styles.mainTitle}>
              Upload Your Fashion Item
            </h2>
            <p style={styles.subtitle}>
              Upload an image of any clothing item and let our AI recommend complementary pieces to complete your perfect outfit
            </p>
          </div>

          <div style={styles.uploadContainer}>
            {!previewUrl ? (
              <div
                onClick={() => fileInputRef.current?.click()}
                style={styles.uploadArea}
                onMouseEnter={(e) => {
                  e.target.style.borderColor = '#a855f7';
                  e.target.style.backgroundColor = 'rgba(196, 132, 252, 0.1)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.borderColor = '#c084fc';
                  e.target.style.backgroundColor = 'transparent';
                }}
              >
                <div>
                  <div style={styles.uploadIcon}>
                    <Upload size={32} color="white" />
                  </div>
                  <div>
                    <p style={styles.uploadText}>
                      Click to upload an image
                    </p>
                    <p style={styles.uploadSubtext}>
                      Supports JPG, PNG, JPEG files
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div style={styles.previewContainer}>
                <div style={styles.previewImage}>
                  <img
                    src={previewUrl}
                    alt="Preview"
                    style={styles.previewImg}
                  />
                </div>
                <div style={styles.buttonContainer}>
                  <button
                    onClick={handleUpload}
                    disabled={loading}
                    style={{
                      ...styles.primaryButton,
                      opacity: loading ? 0.5 : 1
                    }}
                    onMouseEnter={(e) => {
                      if (!loading) {
                        e.target.style.background = 'linear-gradient(to right, #9333ea, #db2777)';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!loading) {
                        e.target.style.background = 'linear-gradient(to right, #a855f7, #ec4899)';
                      }
                    }}
                  >
                    {loading ? (
                      <>
                        <Loader2 size={20} style={{ animation: 'spin 1s linear infinite' }} />
                        Getting Recommendations...
                      </>
                    ) : (
                      <>
                        <Sparkles size={20} />
                        Get Recommendations
                      </>
                    )}
                  </button>
                  <button
                    onClick={resetUpload}
                    disabled={loading}
                    style={{
                      ...styles.secondaryButton,
                      opacity: loading ? 0.5 : 1
                    }}
                    onMouseEnter={(e) => {
                      if (!loading) {
                        e.target.style.backgroundColor = '#f9fafb';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!loading) {
                        e.target.style.backgroundColor = 'white';
                      }
                    }}
                  >
                    Reset
                  </button>
                </div>
              </div>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageSelect}
              style={styles.hiddenInput}
            />
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div style={styles.errorContainer}>
            <div style={styles.errorBox}>
              <X size={20} color="#dc2626" />
              <p style={styles.errorText}>{error}</p>
            </div>
          </div>
        )}

        {/* Results Section */}
        {results && (
          <div style={styles.resultsSection}>
            {/* Detected Item */}
            <div style={styles.detectedSection}>
              <div style={styles.sectionHeader}>
                <Check size={24} color="#10b981" />
                <h3 style={styles.sectionHeaderTitle}>Detected Item</h3>
              </div>
              <div style={styles.metadataGrid}>
                {Object.entries(results.input_metadata).map(([key, value]) => (
                  <div key={key} style={styles.metadataItem}>
                    <p style={styles.metadataLabel}>
                      {key === 'baseColour' ? 'Color' : key === 'subCategory' ? 'Category' : key}
                    </p>
                    <p style={styles.metadataValue}>{value}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Recommendations */}
            <div>
              <div style={styles.sectionHeader}>
                <Heart size={24} color="#ec4899" />
                <h3 style={styles.sectionHeaderTitle}>Recommended Items</h3>
              </div>

              {Object.keys(results.recommendations).length === 0 ? (
                <div style={styles.noResults}>
                  <ShoppingBag size={48} style={styles.noResultsIcon} />
                  <p style={styles.noResultsText}>
                    No complementary items found for this combination. Try uploading a different item!
                  </p>
                </div>
              ) : (
                <div style={styles.recommendationsContainer}>
                  {Object.entries(results.recommendations).map(([category, items]) => (
                    <div key={category}>
                      <h4 style={styles.categoryTitle}>
                        {category}
                      </h4>
                      <div style={styles.recommendationsGrid}>
                        {items.map((item, index) => (
                          <RecommendationCard
                            key={`${category}-${index}`}
                            item={item}
                            category={category}
                          />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div style={styles.footer}>
        <div style={styles.footerContent}>
          <p>Powered by AI Fashion Recommendation System</p>
        </div>
      </div>
    </div>
  );
};

export default FashionRecommendationApp;