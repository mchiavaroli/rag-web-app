/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  async rewrites() {
    return [
      {
        source: '/images/:path*',
        destination: `${process.env.RAG_BACKEND_URL || 'http://localhost:8000'}/images/:path*`,
      },
    ]
  },
}

export default nextConfig
