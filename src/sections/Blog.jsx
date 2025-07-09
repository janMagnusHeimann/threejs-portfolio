import { useState } from 'react';
import { blogPosts } from '../constants/index.js';
import BlogCard from '../components/BlogCard.jsx';
import BlogPost from '../components/BlogPost.jsx';

const Blog = () => {
  const [selectedPost, setSelectedPost] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');

  // Get unique categories
  const categories = ['All', ...new Set(blogPosts.map(post => post.category))];

  // Filter posts based on search and category
  const filteredPosts = blogPosts.filter(post => {
    const matchesSearch = post.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         post.excerpt.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         post.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
    
    const matchesCategory = selectedCategory === 'All' || post.category === selectedCategory;
    
    return matchesSearch && matchesCategory;
  });

  // Sort posts by date (newest first)
  const sortedPosts = filteredPosts.sort((a, b) => new Date(b.date) - new Date(a.date));

  const handleReadMore = (post) => {
    setSelectedPost(post);
  };

  const handleBack = () => {
    setSelectedPost(null);
  };

  if (selectedPost) {
    return <BlogPost post={selectedPost} onBack={handleBack} />;
  }

  return (
    <section className="c-space my-20" id="blog">
      <div className="w-full">
        {/* Header */}
        <div className="text-center mb-12">
          <h2 className="head-text">Tech Blog</h2>
          <p className="text-gray-500 text-lg mt-4 max-w-2xl mx-auto">
            Sharing insights on machine learning, web development, and cutting-edge technology
          </p>
        </div>

        {/* Search and Filter */}
        <div className="mb-8 flex flex-col md:flex-row gap-4 items-center justify-between">
          {/* Search */}
          <div className="relative flex-1 max-w-md">
            <input
              type="text"
              placeholder="Search articles..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-black-300 border border-black-200 rounded-lg px-4 py-2 pl-10 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
            />
            <svg 
              className="absolute left-3 top-2.5 w-5 h-5 text-gray-400" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>

          {/* Category Filter */}
          <div className="flex gap-2 flex-wrap">
            {categories.map(category => (
              <button
                key={category}
                onClick={() => setSelectedCategory(category)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  selectedCategory === category
                    ? 'bg-blue-500 text-white'
                    : 'bg-black-300 text-gray-300 hover:bg-black-200'
                }`}
              >
                {category}
              </button>
            ))}
          </div>
        </div>

        {/* Results count */}
        <div className="mb-6">
          <p className="text-gray-400 text-sm">
            {filteredPosts.length === blogPosts.length 
              ? `${filteredPosts.length} articles` 
              : `${filteredPosts.length} of ${blogPosts.length} articles`}
          </p>
        </div>

        {/* Featured Posts */}
        {searchTerm === '' && selectedCategory === 'All' && (
          <div className="mb-12">
            <h3 className="text-2xl font-bold text-white mb-6">Featured Articles</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {sortedPosts.filter(post => post.featured).map(post => (
                <BlogCard 
                  key={post.id} 
                  post={post} 
                  onReadMore={handleReadMore}
                />
              ))}
            </div>
          </div>
        )}

        {/* All Posts */}
        <div className="mb-12">
          {searchTerm === '' && selectedCategory === 'All' && (
            <h3 className="text-2xl font-bold text-white mb-6">All Articles</h3>
          )}
          
          {filteredPosts.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-gray-400 text-lg">
                No articles found matching your search.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {sortedPosts.map(post => (
                <BlogCard 
                  key={post.id} 
                  post={post} 
                  onReadMore={handleReadMore}
                />
              ))}
            </div>
          )}
        </div>

        {/* Newsletter Signup */}
        <div className="bg-black-300 rounded-lg p-8 text-center border border-black-200">
          <h3 className="text-2xl font-bold text-white mb-4">Stay Updated</h3>
          <p className="text-gray-400 mb-6">
            Get notified when I publish new articles about machine learning, web development, and technology insights.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 max-w-md mx-auto">
            <input
              type="email"
              placeholder="Enter your email"
              className="flex-1 bg-black-200 border border-black-100 rounded-lg px-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
            />
            <button className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg font-medium transition-colors">
              Subscribe
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Blog;